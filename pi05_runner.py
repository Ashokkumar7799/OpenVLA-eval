"""
Physical AI: Pi0.5 (π₀.₅) LeRobot Fine-Tuning & MuJoCo Evaluation Runner
========================================================================
Integrates Physical Intelligence's π₀.₅ (Pi05) Foundation Model with:
  1. Fine-tuning on lerobot/aloha_static_screw_driver via LeRobot (bfloat16, train_expert_only).
  2. High-speed 50 Hz closed-loop Action Chunking inference inside MuJoCo Digital Twin.
  3. Proper float32 [0.0, 1.0] visual scaling, calibrated gripper proprioception,
     and Sim-to-Real camera/gripper visual alignment for frozen PaliGemma VLM.
  4. Out-of-Distribution (OOD) Red stress testing.
"""

import os
import sys
if sys.platform != "win32" and "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "mujoco_menagerie", "aloha", "scene_with_screwdriver.xml")
SPLITS_JSON = os.path.join(SCRIPT_DIR, "dataset_splits.json")
LOCAL_DATASET_DIR = r"E:\mediapipe\robotics_datasets\aloha_static_screw_driver"

# Normalization bounds for ALOHA 14-DoF action space
ACTION_MIN = np.array([
    -0.0276, -0.9986,  0.9311, -0.4126, -1.1950, -0.2286, -0.0092,
    -0.2807, -0.9710, -0.0107, -0.1488, -0.9403, -0.3007, -0.0834
], dtype=np.float32)

ACTION_MAX = np.array([
     0.5354,  0.0046,  1.3530,  0.1227, -0.3037,  0.4034,  0.7713,
     0.2255,  0.5216,  1.1796,  0.1089,  0.7194,  0.3252,  0.7697
], dtype=np.float32)

# Empirical ALOHA gripper calibration from dataset stats (99th percentile / max):
# Action gripper commands span ~[0.0, 0.75..0.77] rad; State gripper feedback spans ~[0.0, 0.84..0.86] rad.
ACTION_GRIPPER_OPEN_RAD = 0.75
STATE_GRIPPER_OPEN_RAD = 0.84


def parse_args():
    parser = argparse.ArgumentParser(description="Pi0.5 Aloha Runner (Training & MuJoCo Rollout)")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval",
                        help="'train' (launches LeRobot Pi0.5 fine-tuning) or 'eval' (MuJoCo Digital Twin rollout)")
    parser.add_argument("--policy_path", type=str, default="./outputs/pi05_aloha_expert/checkpoints/last/pretrained_model",
                        help="Path to trained Pi0.5 checkpoint or Hugging Face repo")
    parser.add_argument("--episodes", type=int, nargs="+", default=[35],
                        help="Holdout test episode IDs to evaluate (e.g. --episodes 35 36 37)")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps for LeRobot train (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for single 24GB L4 GPU (default: 8)")
    parser.add_argument("--exec_steps", type=int, default=25,
                        help="Action steps executed per predicted chunk before replanning (default: 25 = 0.5s closed-loop)")
    parser.add_argument("--no_visual_align", action="store_true",
                        help="Disable Sim-to-Real camera FOV/cam_low/white-gripper visual alignment")
    parser.add_argument("--shadow_size", type=int, default=1024,
                        help="MuJoCo shadow map resolution (1024 is ~2.4x faster than 8192; set 8192 for legacy)")
    parser.add_argument("--ood_red", action="store_true", help="OOD Stress Test: Recolors tool & cup to RED")
    parser.add_argument("--save_video", action="store_true", help="Record and save MP4 rollout video")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi05_aloha_expert", help="Output directory")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--task", type=str,
                        default="Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup.",
                        help="Natural language task instruction for Pi0.5 VLA policy")
    return parser.parse_args()


def scale_gripper(a_grip):
    norm = np.clip(a_grip / ACTION_GRIPPER_OPEN_RAD, 0.0, 1.0)
    return 0.002 + norm * (0.037 - 0.002)


def mujoco_to_gripper_rad(pos_m):
    norm = np.clip((pos_m - 0.002) / (0.037 - 0.002), 0.0, 1.0)
    return float(norm * STATE_GRIPPER_OPEN_RAD)


def map_action_to_ctrl(act):
    ctrl = np.zeros(14, dtype=np.float32)
    ctrl[0:6] = act[0:6]
    ctrl[6] = scale_gripper(act[6])
    ctrl[7:13] = act[7:13]
    ctrl[13] = scale_gripper(act[13])
    return ctrl


def apply_visual_alignment(model, shadow_size=1024):
    """
    Aligns MuJoCo visual rendering (cameras, white 3D-printed grippers, studio backdrop)
    with the real ALOHA training dataset without modifying any collision geometry or physics.
    Essential because Pi0.5 fine-tuning freezes the PaliGemma vision encoder.
    """
    import mujoco

    # 1. White 3D-printed ALOHA finger wings & wrist mounts (matches real dataset grippers)
    for g_id in range(model.ngeom):
        m_id = model.geom_dataid[g_id] if model.geom_type[g_id] == mujoco.mjtGeom.mjGEOM_MESH else -1
        if m_id >= 0:
            m_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m_id) or ""
            if any(k in m_name for k in ["custom_finger", "gripper_wrist_mount"]):
                model.geom_matid[g_id] = -1
                model.geom_rgba[g_id] = [0.84, 0.85, 0.87, 1.0]

    # 2. Neutral studio floor & light-tent backdrop (removes blue checkerboard & blue skybox)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_gid != -1:
        model.geom_matid[floor_gid] = -1
        model.geom_rgba[floor_gid] = [0.68, 0.69, 0.71, 1.0]
        model.geom_size[floor_gid] = [8.0, 8.0, 0.05]

    model.vis.rgba.haze[:] = [0.80, 0.81, 0.82, 1.0]
    if model.ntex > 0:
        for t_id in range(model.ntex):
            if model.tex_type[t_id] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                adr = model.tex_adr[t_id]
                w, h = model.tex_width[t_id], model.tex_height[t_id]
                nchan = model.tex_nchannel[t_id]
                sz = w * h * nchan
                model.tex_data[adr:adr + sz] = 215

    # 3. Tight cam_high FOV matching real 640x480 D405 overhead crop (black table fills frame)
    cid_high = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
    if cid_high != -1:
        model.cam_sensorsize[cid_high] = [0.0, 0.0]
        model.cam_fovy[cid_high] = 35.5
        ang = np.radians(15.5)
        model.cam_quat[cid_high] = [np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0]

    # 4. Wrist cams FOV matching real D405 wrist crop
    for wc in ["wrist_cam_left", "wrist_cam_right"]:
        cid_w = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, wc)
        if cid_w != -1:
            model.cam_sensorsize[cid_w] = [0.0, 0.0]
            model.cam_fovy[cid_w] = 48.0

    # 5. Calibrate cam_low (worms_eye_cam) -> rotated 90 deg CCW view matching real dataset cam_low
    cid_low = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "worms_eye_cam")
    if cid_low != -1:
        model.cam_sensorsize[cid_low] = [0.0, 0.0]
        model.cam_fovy[cid_low] = 48.0
        model.cam_pos[cid_low] = [-0.03, -0.34, 0.56]
        target = np.array([-0.03, 0.25, 0.0])
        fwd = target - model.cam_pos[cid_low]
        fwd = fwd / np.linalg.norm(fwd)
        up_hint = np.array([1.0, 0.0, 0.0])  # +X points UP in image (90 deg CCW)
        z_cam = -fwd
        x_cam = np.cross(up_hint, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= np.linalg.norm(y_cam)
        R = np.column_stack([x_cam, y_cam, z_cam])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        model.cam_quat[cid_low] = q

    model.vis.quality.shadowsize = int(shadow_size)


def calibrate_screwdriver_contact(model, torsional_friction=0.010):
    """Lower torsional spin resistance on the screwdriver handle/shaft/tip so
    gravity gently levels the screwdriver inside the right pinch grasp instead
    of locking it steeply nose-up into handover."""
    import mujoco
    for gname in ("screwdriver_grip", "screwdriver_shaft", "screwdriver_tip"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid != -1:
            model.geom_friction[gid, 1] = float(torsional_friction)


def compute_episode_grasp_point(model, data, actions, x_offset=-0.012):
    import mujoco
    s_finger_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/left_finger")
    s_finger_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/right_finger")
    use_site = (s_finger_l != -1 and s_finger_r != -1)
    if not use_site:
        b_finger_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/left_finger_link")
        b_finger_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/right_finger_link")
    z_list = []
    pos_list = []
    for f in range(min(220, len(actions))):
        act = actions[f]
        data.qpos[0:6] = act[0:6]
        data.qpos[6] = data.qpos[7] = scale_gripper(act[6])
        data.qpos[8:14] = act[7:13]
        data.qpos[14] = data.qpos[15] = scale_gripper(act[13])
        mujoco.mj_forward(model, data)
        if use_site:
            pt = (data.site_xpos[s_finger_l] + data.site_xpos[s_finger_r]) / 2.0
        else:
            pt = (data.xpos[b_finger_l] + data.xpos[b_finger_r]) / 2.0
        z_list.append(pt[2])
        pos_list.append(pt.copy())

    min_f = int(np.argmin(z_list))
    gx = float(pos_list[min_f][0]) + float(x_offset)
    gy = float(pos_list[min_f][1])
    return gx, gy


def run_pi05_training(args):
    """Launches LeRobot Pi0.5 fine-tuning sized for a single 24GB L4 GPU."""
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
    is_explicit_none = policy_path.lower() in {"none", "gt", "ground_truth"}
    if not is_explicit_none and not os.path.exists(policy_path) and not ("/" in policy_path and not policy_path.startswith((".", "/"))):
        # Auto-detect latest numeric checkpoint if exact path not found (e.g. 002000/pretrained_model)
        candidates = glob.glob(os.path.join(args.output_dir, "**", "pretrained_model"), recursive=True)
        if candidates:
            candidates.sort()
            policy_path = candidates[-1]
            print(f"ℹ️ Auto-detected latest checkpoint: {policy_path}")

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
                preprocessor_overrides={"device_processor": {"device": str(device)}}
            )
            print(f"✅ Policy checkpoint & pre/post-processors loaded successfully from '{policy_path}'!")
        except Exception as e:
            print(f"⚠️ Warning: Could not instantiate policy from {policy_path} ({e}).")
            policy = None
    else:
        print(f"ℹ️ No local checkpoint or repo loaded ('{policy_path}').")
        print("   Running Ground-Truth Expert trajectory inside calibrated MuJoCo Digital Twin.")

    # Load telemetry (prefer local dataset if available so offline checks work cleanly)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if os.path.exists(LOCAL_DATASET_DIR):
        repo_dir = LOCAL_DATASET_DIR
    else:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(
            "lerobot/aloha_static_screw_driver",
            repo_type="dataset",
            allow_patterns=["*.parquet", "meta/**"],
            token=hf_token
        )
    parquet_files = sorted(
        p for p in glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True)
        if "meta" not in p
    )
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)

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

        # Initialize Simulation (qpos[6] & qpos[14] match single-finger rail coordinates directly)
        mujoco.mj_resetData(model, data)
        data.qpos[0:6] = initial_state[0:6]
        data.qpos[6] = data.qpos[7] = scale_gripper(initial_state[6])
        data.qpos[8:14] = initial_state[7:13]
        data.qpos[14] = data.qpos[15] = scale_gripper(initial_state[13])
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
                gt_act = gt_actions[min(f_idx, len(gt_actions)-1)]

                if policy is not None:
                    if len(action_buffer) == 0:
                        # 1. Capture 4 camera views scaled to float32 [0.0, 1.0]
                        t_high, img_high = _render_01_tensor("overhead_cam")
                        t_low, _ = _render_01_tensor("worms_eye_cam")
                        t_wl, _ = _render_01_tensor("wrist_cam_left")
                        t_wr, _ = _render_01_tensor("wrist_cam_right")

                        # 2. Extract current 14-DoF proprioceptive joint state
                        curr_state = np.zeros(14, dtype=np.float32)
                        curr_state[0:6] = data.qpos[0:6]
                        curr_state[6] = mujoco_to_gripper_rad(data.qpos[6])
                        curr_state[7:13] = data.qpos[8:14]
                        curr_state[13] = mujoco_to_gripper_rad(data.qpos[14])

                        # 3. Build observation batch matching training preprocessor expectations
                        raw_obs = {
                            "observation.images.cam_high": t_high,
                            "observation.images.cam_low": t_low,
                            "observation.images.cam_left_wrist": t_wl,
                            "observation.images.cam_right_wrist": t_wr,
                            "observation.state": torch.from_numpy(curr_state).unsqueeze(0),
                            "task": [args.task],
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
        finally:
            if renderer is not None:
                renderer.close()

        elapsed = time.time() - t_start
        lift_height_cm = float((max_z - initial_z) * 100.0)
        lift_success = bool(lift_height_cm >= 5.0)

        # Cup placement check
        final_screw_pos = data.xpos[screw_body_id] if screw_body_id != -1 else np.zeros(3)
        cup_pos = model.body_pos[cup_body_id] if cup_body_id != -1 else np.array([-0.12, 0.19, 0.001])
        cup_dist_cm = float(np.linalg.norm(final_screw_pos[:2] - cup_pos[:2]) * 100.0)
        cup_success = bool(cup_dist_cm < 10.0 and final_screw_pos[2] < 0.19)

        mean_l1 = float(np.mean(l1_errors)) if l1_errors else 0.0
        mode_str = "🤖 Pi0.5 Autonomous Policy" if policy is not None else "📋 Human Ground-Truth Replay"
        print(f"\n{mode_str} Rollout Completed in {elapsed:.2f} seconds!")
        print(f"📊 Results: Lift={lift_success} ({lift_height_cm:.1f} cm), Cup={cup_success} (Dist={cup_dist_cm:.1f} cm), Mean L1 Err={mean_l1:.4f} rad")

        ood_suffix = "_ood_red" if args.ood_red else ""
        if args.save_video and len(frames) > 0:
            v_path = os.path.join(args.output_dir, f"eval_pi05_episode_{ep_id}{ood_suffix}.mp4")
            try:
                import mediapy as media
                media.write_video(v_path, frames, fps=50)
            except Exception:
                import cv2
                h_v, w_v = frames[0].shape[:2]
                vw = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (w_v, h_v))
                for fr in frames:
                    vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                vw.release()
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
