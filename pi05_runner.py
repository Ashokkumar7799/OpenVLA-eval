"""
Physical AI: Pi0.5 (π₀.₅) LeRobot Fine-Tuning & MuJoCo Evaluation Runner
========================================================================
Integrates Physical Intelligence's π₀.₅ (Pi05) Foundation Model with:
  1. Fine-tuning on lerobot/aloha_static_screw_driver via LeRobot (bfloat16, train_expert_only).
  2. High-speed 50 Hz Action Chunking inference (~10 seconds per episode rollout).
  3. Closed-loop evaluation inside our calibrated MuJoCo Digital Twin (scene_with_screwdriver.xml).
  4. Out-of-Distribution (OOD) Red stress testing.
"""

import os
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
import sys
import shutil
import shlex
import subprocess
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "mujoco_menagerie", "aloha", "scene_with_screwdriver.xml")
SPLITS_JSON = os.path.join(SCRIPT_DIR, "dataset_splits.json")

# Normalization bounds for ALOHA 14-DoF action space
ACTION_MIN = np.array([
    -0.0276, -0.9986,  0.9311, -0.4126, -1.1950, -0.2286, -0.0092,
    -0.2807, -0.9710, -0.0107, -0.1488, -0.9403, -0.3007, -0.0834
], dtype=np.float32)

ACTION_MAX = np.array([
     0.5354,  0.0046,  1.3530,  0.1227, -0.3037,  0.4034,  0.7713,
     0.2255,  0.5216,  1.1796,  0.1089,  0.7194,  0.3252,  0.7697
], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="Pi0.5 Aloha Runner (Training & MuJoCo Rollout)")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval",
                        help="'train' (launches LeRobot Pi0.5 fine-tuning) or 'eval' (MuJoCo Digital Twin rollout)")
    parser.add_argument("--policy_path", type=str, default="./outputs/pi05_aloha_expert/checkpoints/last/pretrained_model",
                        help="Path to trained Pi0.5 checkpoint or Hugging Face repo (default: lerobot/pi05_base)")
    parser.add_argument("--episodes", type=int, nargs="+", default=[35],
                        help="Holdout test episode IDs to evaluate (e.g. --episodes 35 36 37)")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps for LeRobot train (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for single 24GB L4 GPU (default: 8)")
    parser.add_argument("--ood_red", action="store_true", help="OOD Stress Test: Recolors tool & cup to RED")
    parser.add_argument("--save_video", action="store_true", help="Record and save MP4 rollout video")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi05_aloha_expert", help="Output directory")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--task", type=str,
                        default="Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup.",
                        help="Natural language task instruction for Pi0.5 VLA policy")
    return parser.parse_args()

def scale_gripper(a_grip):
    norm = np.clip(a_grip / 0.69, 0.0, 1.0)
    return 0.002 + norm * (0.037 - 0.002)

def mujoco_to_gripper_rad(pos_m):
    norm = np.clip((pos_m - 0.002) / (0.037 - 0.002), 0.0, 1.0)
    return float(norm * 0.69)

def map_action_to_ctrl(act):
    ctrl = np.zeros(14, dtype=np.float32)
    ctrl[0:6] = act[0:6]
    ctrl[6] = scale_gripper(act[6])
    ctrl[7:13] = act[7:13]
    ctrl[13] = scale_gripper(act[13])
    return ctrl

def compute_episode_grasp_point(model, data, actions):
    import mujoco
    b_finger_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/left_finger_link")
    b_finger_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/right_finger_link")
    z_list = []
    pos_list = []
    for f in range(min(220, len(actions))):
        act = actions[f]
        data.qpos[0:6] = act[0:6]
        data.qpos[6] = data.qpos[7] = scale_gripper(act[6]) / 2.0
        data.qpos[8:14] = act[7:13]
        data.qpos[14] = data.qpos[15] = scale_gripper(act[13]) / 2.0
        mujoco.mj_forward(model, data)
        pt = (data.xpos[b_finger_l] + data.xpos[b_finger_r]) / 2.0
        z_list.append(pt[2])
        pos_list.append(pt)

    min_f = int(np.argmin(z_list))
    gx = float(pos_list[min_f][0])
    gy = float(pos_list[min_f][1])
    return gx, gy

def run_pi05_training(args):
    """Launches LeRobot Pi0.5 fine-tuning sized for a single 24GB L4 GPU."""
    import shutil
    import shlex
    cmd = [
        "lerobot-train",
        "--dataset.repo_id=lerobot/aloha_static_screw_driver",
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
        "--save_freq=1000"
    ]
    train_bin = shutil.which("lerobot-train")
    if train_bin is None:
        cmd = [sys.executable, "-m", "lerobot.scripts.train"] + cmd[1:]
    else:
        cmd[0] = train_bin

    print(f"\n🚀 Launching LeRobot Pi0.5 Fine-Tuning on single 24GB GPU...")
    print(f"Command: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_pi05_evaluation(args):
    """Evaluates Pi0.5 policy inside calibrated MuJoCo Digital Twin with 50 Hz Action Chunking."""
    import mujoco
    import cv2
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔍 Loading Calibrated MuJoCo Digital Twin: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    if args.ood_red:
        for mat_name, rgba in [
            ("screwdriver_handle", [0.88, 0.14, 0.14, 1.0]),
            ("cup_blue_outer", [0.82, 0.12, 0.12, 1.0]),
            ("cup_sponge_yellow", [0.78, 0.20, 0.20, 1.0]),
        ]:
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
            if mid != -1:
                model.mat_rgba[mid] = rgba
        print("🎨 MuJoCo Digital Twin recolored to matching OOD RED (Red Screwdriver + Red Cup).")

    print(f"🤖 Loading Policy: {args.policy_path}")
    policy = None
    preprocess, postprocess = None, None

    policy_path = args.policy_path
    if not os.path.exists(policy_path) and not ("/" in policy_path and not policy_path.startswith((".", "/"))):
        # Auto-detect latest numeric checkpoint if exact path not found (e.g. 002000/pretrained_model)
        candidates = glob.glob(os.path.join(args.output_dir, "**", "pretrained_model"), recursive=True)
        if candidates:
            candidates.sort()
            policy_path = candidates[-1]
            print(f"ℹ️ Auto-detected latest checkpoint: {policy_path}")

    is_hf_repo = "/" in policy_path and not os.path.exists(policy_path) and not policy_path.startswith((".", "/"))
    if os.path.exists(policy_path) or is_hf_repo:
        try:
            from lerobot.policies.pi05 import PI05Policy
            from lerobot.policies.factory import make_pre_post_processors

            policy = PI05Policy.from_pretrained(policy_path).to(device)
            policy.eval()
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}}
            )
            print(f"✅ Policy checkpoint & pre/post-processors loaded successfully from '{policy_path}'!")
        except Exception as e:
            print(f"⚠️ Warning: Could not instantiate policy from {policy_path} ({e}).")
            policy = None
    else:
        print(f"ℹ️ No local checkpoint or repo found at '{policy_path}'.")
        print("   Running Ground-Truth Expert trajectory inside calibrated MuJoCo Digital Twin.")

    # Load telemetry to obtain human expert grasp reference point
    from huggingface_hub import snapshot_download
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    repo_dir = snapshot_download(
        "lerobot/aloha_static_screw_driver",
        repo_type="dataset",
        allow_patterns=["*.parquet", "meta/**"],
        token=hf_token
    )
    parquet_files = glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True)
    df = pd.concat([pd.read_parquet(p) for p in parquet_files if "meta" not in p], ignore_index=True)

    results = []
    print(f"\n⚡ Running Evaluation on Episodes: {args.episodes}")

    for ep_id in args.episodes:
        ep_data = df[df["episode_index"] == ep_id].sort_values("frame_index")
        gt_actions = np.array([r["action"] for _, r in ep_data.iterrows()], dtype=np.float32)
        initial_state = gt_actions[0]

        gx, gy = compute_episode_grasp_point(model, data, gt_actions)
        print(f"\n--- Episode {ep_id} (Grasp Origin: X={gx:.4f}, Y={gy:.4f}) ---")

        # Reset policy action buffer and visual memory for clean episode rollout
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()

        # Initialize Simulation
        mujoco.mj_resetData(model, data)
        data.qpos[0:6] = initial_state[0:6]
        data.qpos[6] = data.qpos[7] = scale_gripper(initial_state[6]) / 2.0
        data.qpos[8:14] = initial_state[7:13]
        data.qpos[14] = data.qpos[15] = scale_gripper(initial_state[13]) / 2.0
        data.ctrl[:14] = map_action_to_ctrl(initial_state)

        screw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
        screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
        cup_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")

        if screw_joint_id != -1:
            qadr = model.jnt_qposadr[screw_joint_id]
            data.qpos[qadr:qadr+7] = [gx, gy, 0.015, 0.7071, 0.0, 0.7071, 0.0]

        mujoco.mj_forward(model, data)

        initial_z = data.xpos[screw_body_id][2] if screw_body_id != -1 else 0.0
        max_z = initial_z
        t_start = time.time()

        total_steps = 400
        frames = []
        renderer = mujoco.Renderer(model, height=480, width=640)
        l1_errors = []

        chunk_size = getattr(policy.config, "n_action_steps", 50) if policy is not None else 1
        action_buffer = []

        pbar = tqdm(range(total_steps), desc=f"🤖 MuJoCo Rollout Ep {ep_id}", unit="frame")
        for f_idx in pbar:
            gt_act = gt_actions[min(f_idx, len(gt_actions)-1)]

            if policy is not None:
                if len(action_buffer) == 0:
                    # 1. Capture 4 camera views from Digital Twin for policy input
                    renderer.update_scene(data, camera="overhead_cam")
                    cam_high_img = renderer.render().copy()

                    renderer.update_scene(data, camera="worms_eye_cam")
                    cam_low_img = renderer.render().copy()

                    renderer.update_scene(data, camera="wrist_cam_left")
                    cam_wl_img = renderer.render().copy()

                    renderer.update_scene(data, camera="wrist_cam_right")
                    cam_wr_img = renderer.render().copy()

                    # 2. Extract current 14-DoF proprioceptive joint state with calibrated gripper conversion
                    curr_state = np.zeros(14, dtype=np.float32)
                    curr_state[0:6] = data.qpos[0:6]
                    curr_state[6] = mujoco_to_gripper_rad(data.qpos[6])
                    curr_state[7:13] = data.qpos[8:14]
                    curr_state[13] = mujoco_to_gripper_rad(data.qpos[14])

                    # 3. Build observation batch
                    raw_obs = {
                        "observation.images.cam_high": torch.from_numpy(cam_high_img).permute(2, 0, 1).unsqueeze(0),
                        "observation.images.cam_low": torch.from_numpy(cam_low_img).permute(2, 0, 1).unsqueeze(0),
                        "observation.images.cam_left_wrist": torch.from_numpy(cam_wl_img).permute(2, 0, 1).unsqueeze(0),
                        "observation.images.cam_right_wrist": torch.from_numpy(cam_wr_img).permute(2, 0, 1).unsqueeze(0),
                        "observation.state": torch.from_numpy(curr_state).unsqueeze(0),
                        "task": [args.task],
                        "complementary_data": {
                            "task": [args.task]
                        }
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
                        if isinstance(pred_act, torch.Tensor):
                            pred_act = pred_act.cpu().numpy()
                        else:
                            pred_act = np.array(pred_act)
                    else:
                        if isinstance(action_out, torch.Tensor):
                            pred_act = action_out.cpu().numpy()
                        else:
                            pred_act = np.array(action_out)

                    pred_act = pred_act.squeeze()
                    if pred_act.ndim == 1:
                        pred_act = pred_act[None, :]
                    action_buffer = [row[:14] for row in pred_act[:chunk_size]]

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

            # Step MuJoCo physics with action at exact 50 Hz (10 substeps * 0.002s = 0.020s)
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

        elapsed = time.time() - t_start
        lift_height_cm = float((max_z - initial_z) * 100.0)
        lift_success = bool(lift_height_cm >= 5.0)

        # Cup placement check
        final_screw_pos = data.xpos[screw_body_id] if screw_body_id != -1 else np.zeros(3)
        cup_pos = model.body_pos[cup_body_id] if cup_body_id != -1 else np.array([-0.12, 0.19, 0.001])
        cup_dist_cm = float(np.linalg.norm(final_screw_pos[:2] - cup_pos[:2]) * 100.0)
        cup_success = bool(cup_dist_cm < 10.0 and final_screw_pos[2] < 0.16)

        mean_l1 = float(np.mean(l1_errors)) if l1_errors else 0.0
        mode_str = "🤖 Pi0.5 Autonomous Policy" if policy is not None else "📋 Human Ground-Truth Replay"
        print(f"\n{mode_str} Rollout Completed in {elapsed:.2f} seconds!")
        print(f"📊 Results: Lift={lift_success} ({lift_height_cm:.1f} cm), Cup={cup_success} (Dist={cup_dist_cm:.1f} cm), Mean L1 Err={mean_l1:.4f} rad")

        ood_suffix = "_ood_red" if args.ood_red else ""
        if renderer and args.save_video:
            v_path = os.path.join(args.output_dir, f"eval_pi05_episode_{ep_id}{ood_suffix}.mp4")
            import mediapy as media
            media.write_video(v_path, frames, fps=50)
            print(f"🎥 Saved rollout video -> {v_path}")
            renderer.close()

        results.append({
            "episode": int(ep_id),
            "mode": "autonomous" if policy is not None else "ground_truth",
            "ood_red": bool(args.ood_red),
            "rollout_time_sec": float(elapsed),
            "lift_success": bool(lift_success),
            "lift_height_cm": float(lift_height_cm),
            "cup_success": bool(cup_success),
            "cup_dist_cm": float(cup_dist_cm),
            "mean_l1_error_rad": float(mean_l1)
        })

    res_suffix = "_ood_red" if args.ood_red else ""
    out_json = os.path.join(args.output_dir, f"pi05_benchmark_results{res_suffix}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Benchmark results saved to: {out_json}")

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_pi05_training(args)
    else:
        run_pi05_evaluation(args)
