"""Pi0.5 (π₀.₅) LeRobot fine-tuning launcher and 50 Hz closed-loop MuJoCo evaluation engine."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import numpy as np
from tqdm import tqdm

from .data import load_aloha_dataset, resolve_policy_path, write_video_mp4
from .paths import DEFAULT_HF_REPO, SCENE_XML
from .sim_twin import (
    ACTION_MAX,
    ACTION_MIN,
    DEFAULT_TASK,
    apply_ood_red_theme,
    apply_visual_alignment,
    calibrate_screwdriver_contact,
    compute_episode_grasp_point,
    evaluate_episode_outcome,
    map_action_to_ctrl,
    read_aloha_state_14,
    reset_aloha_episode,
)


def run_pi05_training(args):
    """Launches LeRobot Pi0.5 fine-tuning sized for a single 24GB L4 GPU."""
    dataset_repo = getattr(args, "dataset_repo", DEFAULT_HF_REPO)
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_repo}",
        "--policy.type=pi05",
        "--policy.pretrained_path=lerobot/pi05_base",
        "--policy.push_to_hub=false",
        "--policy.repo_id=local/pi05_aloha_expert",
        '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        "--policy.device=cuda",
        f"--output_dir={os.path.join(args.output_dir, 'checkpoints')}",
        "--job_name=pi05_aloha_expert",
        f"--batch_size={args.batch_size}",
        "--num_workers=4",
        f"--steps={args.steps}",
        "--save_freq=1000",
    ]
    train_bin = shutil.which("lerobot-train")
    if train_bin is None:
        cmd = [sys.executable, "-m", "lerobot.scripts.train"] + cmd[1:]
    else:
        cmd[0] = train_bin

    print("\n🚀 Launching LeRobot Pi0.5 Fine-Tuning on single 24GB GPU...")
    print(f"Command: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_pi05_evaluation(args):
    """Evaluates Pi0.5 policy inside calibrated MuJoCo Digital Twin with 50 Hz Action Chunking."""
    import mujoco

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        torch = None
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🔍 Loading Calibrated MuJoCo Digital Twin: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    calibrate_screwdriver_contact(model)

    if not args.no_visual_align:
        apply_visual_alignment(model, shadow_size=args.shadow_size)
        print(f"📷 Visual Sim-to-Real Alignment enabled (shadowsize={args.shadow_size}).")
    else:
        model.vis.quality.shadowsize = int(args.shadow_size)
        print(f"📷 Visual Sim-to-Real Alignment disabled (raw XML, shadowsize={args.shadow_size}).")

    if args.ood_red:
        apply_ood_red_theme(model)
        print("🎨 MuJoCo Digital Twin recolored to matching OOD RED (Red Screwdriver + Red Cup).")

    print(f"🤖 Loading Policy: {args.policy_path}")
    policy = None
    preprocess, postprocess = None, None

    policy_path = resolve_policy_path(args.policy_path, output_dir=args.output_dir)
    if policy_path != args.policy_path:
        print(f"ℹ️ Auto-detected latest checkpoint: {policy_path}")

    is_explicit_none = policy_path.lower() in {"none", "gt", "ground_truth"}
    is_hf_repo = (not is_explicit_none) and ("/" in policy_path and not os.path.exists(policy_path) and not policy_path.startswith((".", "/")))
    if (not is_explicit_none) and torch is not None and (os.path.exists(policy_path) or is_hf_repo):
        try:
            from lerobot.policies.pi05 import PI05Policy
            from lerobot.policies.factory import make_pre_post_processors

            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            policy = PI05Policy.from_pretrained(policy_path, token=hf_token).to(device)
            policy.eval()
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            print(f"✅ Policy checkpoint & pre/post-processors loaded successfully from '{policy_path}'!")
        except Exception as e:
            print(f"⚠️ Warning: Could not instantiate policy from {policy_path} ({e}).")
            policy = None
    else:
        print(f"ℹ️ No local checkpoint or repo loaded ('{policy_path}').")
        print("   Running Ground-Truth Expert trajectory inside calibrated MuJoCo Digital Twin.")

    df, _ = load_aloha_dataset(hf_token=args.hf_token, need_videos=False)
    screw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
    screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
    cup_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")

    results = []
    print(f"\n⚡ Running Evaluation on Episodes: {args.episodes}")

    for ep_id in args.episodes:
        ep_data = df[df["episode_index"] == ep_id].sort_values("frame_index")
        gt_actions = np.array([r["action"] for _, r in ep_data.iterrows()], dtype=np.float32)
        if "observation.state" in ep_data.columns:
            initial_state = np.array(ep_data["observation.state"].iloc[0], dtype=np.float32)
        else:
            initial_state = gt_actions[0]

        gx, gy = compute_episode_grasp_point(model, data, gt_actions)
        print(f"\n--- Episode {ep_id} (Grasp Origin: X={gx:.4f}, Y={gy:.4f}) ---")

        if policy is not None and hasattr(policy, "reset"):
            policy.reset()

        reset_aloha_episode(model, data, initial_state, gx, gy, screw_joint_id)

        initial_z = data.xpos[screw_body_id][2] if screw_body_id != -1 else 0.0
        max_z = initial_z
        t_start = time.time()

        total_steps = 400
        frames = []
        renderer = mujoco.Renderer(model, height=480, width=640) if (policy is not None or args.save_video) else None
        l1_errors = []

        max_chunk = getattr(policy.config, "n_action_steps", 50) if policy is not None else 1
        exec_steps = max(1, min(int(args.exec_steps), int(max_chunk))) if policy is not None else 1
        action_buffer = []

        def _render_01_tensor(cam_name):
            renderer.update_scene(data, camera=cam_name)
            img_rgb = renderer.render().copy()
            # CRITICAL: LeRobotDataset returns float32 in [0.0, 1.0]. Passing uint8 [0..255]
            # causes resize_with_pad_torch() to clamp every pixel >= 1 to solid white 1.0!
            return torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0), img_rgb

        try:
            pbar = tqdm(range(total_steps), desc=f"🤖 MuJoCo Rollout Ep {ep_id}", unit="frame")
            for f_idx in pbar:
                gt_act = gt_actions[min(f_idx, len(gt_actions) - 1)]

                if policy is not None:
                    if len(action_buffer) == 0:
                        t_high, _ = _render_01_tensor("overhead_cam")
                        t_low, _ = _render_01_tensor("worms_eye_cam")
                        t_wl, _ = _render_01_tensor("wrist_cam_left")
                        t_wr, _ = _render_01_tensor("wrist_cam_right")
                        curr_state = read_aloha_state_14(data)

                        raw_obs = {
                            "observation.images.cam_high": t_high,
                            "observation.images.cam_low": t_low,
                            "observation.images.cam_left_wrist": t_wl,
                            "observation.images.cam_right_wrist": t_wr,
                            "observation.state": torch.from_numpy(curr_state).unsqueeze(0),
                            "task": [getattr(args, "task", DEFAULT_TASK)],
                        }

                        if preprocess:
                            processed_obs = preprocess(raw_obs)
                        else:
                            processed_obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw_obs.items()}

                        with torch.no_grad():
                            if hasattr(policy, "predict_action_chunk"):
                                action_out = policy.predict_action_chunk(processed_obs)
                            else:
                                action_out = policy.select_action(processed_obs)

                        if postprocess:
                            pred_act = postprocess(action_out)
                            if isinstance(pred_act, dict):
                                pred_act = pred_act.get("action", pred_act)
                            pred_act = pred_act.detach().cpu().numpy() if isinstance(pred_act, torch.Tensor) else np.array(pred_act)
                        else:
                            pred_act = action_out.detach().cpu().numpy() if isinstance(action_out, torch.Tensor) else np.array(action_out)

                        pred_act = pred_act.squeeze()
                        if pred_act.ndim == 1:
                            pred_act = pred_act[None, :]
                        action_buffer = [row[:14].copy() for row in pred_act[:exec_steps]]

                        if f_idx == 0:
                            pre_imgs, _ = policy._preprocess_images(processed_obs)
                            print(
                                f"  🔎 Frame 0 Diag: _preprocess_images[0] range=[{pre_imgs[0].min():.2f}, {pre_imgs[0].max():.2f}], "
                                f"state_norm range=[{processed_obs['observation.state'].min():.2f}, {processed_obs['observation.state'].max():.2f}]"
                            )
                            print(f"     Pred[0]: {np.round(action_buffer[0], 3)}")
                            print(f"     GT  [0]: {np.round(gt_act, 3)}")

                    act = action_buffer.pop(0)
                    act = np.clip(act[:14], ACTION_MIN, ACTION_MAX)
                    l1_err = float(np.mean(np.abs(act - gt_act)))
                    l1_errors.append(l1_err)

                    if renderer and args.save_video:
                        renderer.update_scene(data, camera="overhead_cam")
                        frames.append(renderer.render().copy())
                else:
                    act = gt_act
                    if renderer and args.save_video:
                        renderer.update_scene(data, camera="overhead_cam")
                        frames.append(renderer.render().copy())

                data.ctrl[:14] = map_action_to_ctrl(act)
                for _ in range(10):
                    mujoco.mj_step(model, data)

                if screw_body_id != -1:
                    cz = data.xpos[screw_body_id][2]
                    if cz > max_z:
                        max_z = cz

                cur_lift = (max_z - initial_z) * 100.0
                last_err = l1_errors[-1] if l1_errors else 0.0
                pbar.set_postfix({"lift_cm": f"{cur_lift:.1f}", "l1_err": f"{last_err:.3f}"})
        finally:
            if renderer is not None:
                renderer.close()

        elapsed = time.time() - t_start
        lift_height_cm, lift_success, cup_dist_cm, cup_success = evaluate_episode_outcome(
            model, data, initial_z, max_z, screw_body_id, cup_body_id
        )

        mean_l1 = float(np.mean(l1_errors)) if l1_errors else 0.0
        mode_str = "🤖 Pi0.5 Autonomous Policy" if policy is not None else "📋 Human Ground-Truth Replay"
        print(f"\n{mode_str} Rollout Completed in {elapsed:.2f} seconds!")
        print(f"📊 Results: Lift={lift_success} ({lift_height_cm:.1f} cm), Cup={cup_success} (Dist={cup_dist_cm:.1f} cm), Mean L1 Err={mean_l1:.4f} rad")

        ood_suffix = "_ood_red" if args.ood_red else ""
        if args.save_video and len(frames) > 0:
            v_path = os.path.join(args.output_dir, f"eval_pi05_episode_{ep_id}{ood_suffix}.mp4")
            write_video_mp4(v_path, frames, fps=50.0)
            print(f"🎥 Saved rollout video -> {v_path}")

        results.append({
            "episode": int(ep_id),
            "mode": "autonomous" if policy is not None else "ground_truth",
            "ood_red": bool(args.ood_red),
            "visual_align": bool(not args.no_visual_align),
            "exec_steps": int(exec_steps),
            "rollout_time_sec": float(elapsed),
            "lift_success": bool(lift_success),
            "lift_height_cm": float(lift_height_cm),
            "cup_success": bool(cup_success),
            "cup_dist_cm": float(cup_dist_cm),
            "mean_l1_error_rad": float(mean_l1),
        })

    res_suffix = "_ood_red" if args.ood_red else ""
    out_json = os.path.join(args.output_dir, f"pi05_benchmark_results{res_suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Benchmark results saved to: {out_json}")
