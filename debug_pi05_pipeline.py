#!/usr/bin/env python3
"""
Pi0.5 (π₀.₅) End-to-End Diagnostic Debugger (Run on Colab GPU in ~30 seconds)
=============================================================================
Systematically isolates model weights, processor normalization, uint8->float
image scaling, MuJoCo visual alignment, and physics GT replay.

Usage:
  python3 debug_pi05_pipeline.py --policy_path ./outputs/pi05_aloha_expert/checkpoints/checkpoints/last/pretrained_model
"""

import os
import sys
if sys.platform != "win32" and "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import glob
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "mujoco_menagerie", "aloha", "scene_with_screwdriver.xml")
TASK_TEXT = "Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup."

ACTION_NAMES = [
    "L_waist", "L_shld", "L_elbow", "L_roll", "L_wristA", "L_wristR", "L_grip",
    "R_waist", "R_shld", "R_elbow", "R_roll", "R_wristA", "R_wristR", "R_grip",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Debug Pi0.5 Evaluation Pipeline")
    parser.add_argument(
        "--policy_path",
        type=str,
        default="./outputs/pi05_aloha_expert/checkpoints/checkpoints/last/pretrained_model",
    )
    parser.add_argument("--episode", type=int, default=35)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--task", type=str, default=TASK_TEXT)
    return parser.parse_args()


def load_dataset_repo(hf_token=None, need_videos=True):
    local_ds = r"E:\mediapipe\robotics_datasets\aloha_static_screw_driver"
    if os.path.exists(local_ds):
        repo_dir = local_ds
    else:
        from huggingface_hub import snapshot_download
        token = hf_token or os.environ.get("HF_TOKEN")
        patterns = ["*.parquet", "meta/**"]
        if need_videos:
            patterns.append("videos/**")
        repo_dir = snapshot_download(
            "lerobot/aloha_static_screw_driver",
            repo_type="dataset",
            allow_patterns=patterns,
            token=token,
        )
    parquet_files = sorted(
        p for p in glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True)
        if "meta" not in p
    )
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    return df, repo_dir


def resolve_policy_path(policy_path):
    if os.path.exists(policy_path):
        return policy_path
    if "/" in policy_path and not policy_path.startswith((".", "/")):
        return policy_path
    candidates = glob.glob(os.path.join("./outputs", "**", "pretrained_model"), recursive=True)
    if candidates:
        candidates.sort()
        return candidates[-1]
    return policy_path


def read_real_4cam_frame(repo_dir, abs_frame_idx):
    import cv2
    cams = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    out = {}
    for c in cams:
        matches = glob.glob(os.path.join(repo_dir, "videos", f"*{c}*", "**", "*.mp4"), recursive=True)
        if not matches:
            return None
        cap = cv2.VideoCapture(matches[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(abs_frame_idx))
        ret, bgr = cap.read()
        cap.release()
        if not ret or bgr is None:
            return None
        out[c] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def main():
    args = parse_args()
    import mujoco
    from pi05_runner import (
        apply_visual_alignment,
        compute_episode_grasp_point,
        map_action_to_ctrl,
        mujoco_to_gripper_rad,
        scale_gripper,
    )

    print("=" * 78)
    print("🔬 Pi0.5 (π₀.₅) SYSTEMATIC DIAGNOSTIC DEBUGGER")
    print("=" * 78)

    df, repo_dir = load_dataset_repo(args.hf_token, need_videos=True)
    ep_df = df[df["episode_index"] == args.episode].sort_values("frame_index")
    gt_actions = np.array([r["action"] for _, r in ep_df.iterrows()], dtype=np.float32)
    gt_states = np.array([r["observation.state"] for _, r in ep_df.iterrows()], dtype=np.float32)
    abs_indices = ep_df["index"].to_numpy()
    print(f"📂 Loaded Episode {args.episode}: {len(ep_df)} frames (abs index {abs_indices[0]}..{abs_indices[-1]})")

    # ------------------------------------------------------------------
    # TEST 1: Local MuJoCo Physics Ground-Truth Replay Check
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("📋 TEST 1: MuJoCo Physics Ground-Truth Replay (50 Hz, 10 substeps)")
    print("-" * 78)
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    gx, gy = compute_episode_grasp_point(model, data, gt_actions)
    screw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
    screw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")

    mujoco.mj_resetData(model, data)
    init = gt_states[0]
    data.qpos[0:6] = init[0:6]
    data.qpos[6] = data.qpos[7] = scale_gripper(init[6])
    data.qpos[8:14] = init[7:13]
    data.qpos[14] = data.qpos[15] = scale_gripper(init[13])
    data.ctrl[:14] = map_action_to_ctrl(init)
    data.qpos[model.jnt_qposadr[screw_jid]:model.jnt_qposadr[screw_jid] + 7] = [
        gx, gy, 0.015, 0.7071, 0.0, 0.7071, 0.0
    ]
    mujoco.mj_forward(model, data)
    init_z = float(data.xpos[screw_bid][2])
    max_z = init_z
    for act in gt_actions:
        data.ctrl[:14] = map_action_to_ctrl(act)
        for _ in range(10):
            mujoco.mj_step(model, data)
        max_z = max(max_z, float(data.xpos[screw_bid][2]))
    cup_pos = model.body_pos[cup_bid]
    cup_dist = float(np.linalg.norm(data.xpos[screw_bid][:2] - cup_pos[:2]) * 100.0)
    print(f"  Grasp Origin: X={gx:.4f}, Y={gy:.4f}")
    print(f"  ✅ GT Lift: {(max_z - init_z) * 100.0:.2f} cm | Final Cup Dist: {cup_dist:.2f} cm")

    # ------------------------------------------------------------------
    # Load Policy + Pre/Post Processors
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        print("\nℹ️ PyTorch not installed in this local env; stopping after Test 1 (GT physics).")
        return

    policy_path = resolve_policy_path(args.policy_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🤖 Loading Policy from '{policy_path}' on {device}...")
    if not (os.path.exists(policy_path) or ("/" in policy_path and not policy_path.startswith((".", "/")))):
        print("❌ No policy checkpoint found. Run this on Colab where checkpoint exists (or pass HF repo).")
        return

    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    policy = PI05Policy.from_pretrained(policy_path, token=args.hf_token).to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"  config.chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}, norm={policy.config.normalization_mapping}")

    def run_chunk(cams_rgb, state_14, scale_to_01=True):
        obs = {}
        for k, img in cams_rgb.items():
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            if scale_to_01:
                t = t.float() / 255.0
            obs[f"observation.images.{k}"] = t
        obs["observation.state"] = torch.from_numpy(state_14.astype(np.float32)).unsqueeze(0)
        obs["task"] = [args.task]
        proc = preprocess(obs)
        # Inspect actual tensor after _preprocess_images!
        pre_imgs, _ = policy._preprocess_images(proc)
        with torch.no_grad():
            chunk_norm = policy.predict_action_chunk(proc)
        chunk_phys = postprocess(chunk_norm)
        if isinstance(chunk_phys, dict):
            chunk_phys = chunk_phys.get("action", chunk_phys)
        chunk_np = chunk_phys.detach().cpu().numpy().squeeze()
        return pre_imgs, chunk_norm.detach().cpu().numpy().squeeze(), chunk_np

    # ------------------------------------------------------------------
    # TEST 2: Real Dataset Frames Open-Loop Check (Teacher-Forced Vision)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("📋 TEST 2: Real Dataset Frames -> Policy Prediction (Validates Trained Weights)")
    print("-" * 78)
    ds_ep = None
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds_ep = LeRobotDataset("lerobot/aloha_static_screw_driver", episodes=[args.episode])
        print(f"  Loaded via LeRobotDataset: {len(ds_ep)} frames (same decoder as lerobot-train)")
    except Exception as e:
        print(f"  Note: LeRobotDataset load failed ({e}), falling back to cv2 video read")

    for f_off in [0, 100, 150, 250]:
        if f_off >= len(gt_actions):
            continue
        if ds_ep is not None and f_off < len(ds_ep):
            sample = ds_ep[f_off]
            real_obs = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim in (1, 3) else v)
                for k, v in sample.items()
                if k.startswith("observation.images.") or k == "observation.state"
            }
            real_obs["task"] = [args.task]
            proc = preprocess(real_obs)
            pre_imgs, _ = policy._preprocess_images(proc)
            with torch.no_grad():
                chunk_norm = policy.predict_action_chunk(proc)
            chunk_phys = postprocess(chunk_norm)
            if isinstance(chunk_phys, dict):
                chunk_phys = chunk_phys.get("action", chunk_phys)
            phys_chunk = chunk_phys.detach().cpu().numpy().squeeze()
        else:
            real_cams = read_real_4cam_frame(repo_dir, abs_indices[f_off])
            if real_cams is None:
                print(f"  ⚠️ Could not decode real video frame at offset {f_off}")
                continue
            pre_imgs, _, phys_chunk = run_chunk(real_cams, gt_states[f_off], scale_to_01=True)

        horizon = min(50, len(gt_actions) - f_off, len(phys_chunk))
        l1_0 = float(np.mean(np.abs(phys_chunk[0, :14] - gt_actions[f_off])))
        l1_50 = float(np.mean(np.abs(phys_chunk[:horizon, :14] - gt_actions[f_off:f_off + horizon])))
        print(
            f"  Frame {f_off:3d}: _preprocess_images[0] range=[{pre_imgs[0].min():.2f}, {pre_imgs[0].max():.2f}] | "
            f"Step0 L1={l1_0:.4f} rad | 50-Step Chunk L1={l1_50:.4f} rad"
        )
        if f_off == 150:
            print("    GT[150]  :", np.round(gt_actions[150], 3))
            print("    Pred[150]:", np.round(phys_chunk[0, :14], 3))

    # ------------------------------------------------------------------
    # TEST 3: Prove OLD uint8 Bug vs FIXED float32/255 on MuJoCo Frame 0
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("📋 TEST 3: MuJoCo Frame 0 — OLD uint8 Bug vs FIXED float32 [0,1] vs Visual-Aligned")
    print("-" * 78)

    def render_mj_4cam(mj_model, mj_data, state_14):
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[0:6] = state_14[0:6]
        mj_data.qpos[6] = mj_data.qpos[7] = scale_gripper(state_14[6])
        mj_data.qpos[8:14] = state_14[7:13]
        mj_data.qpos[14] = mj_data.qpos[15] = scale_gripper(state_14[13])
        mj_data.ctrl[:14] = map_action_to_ctrl(state_14)
        mj_data.qpos[mj_model.jnt_qposadr[screw_jid]:mj_model.jnt_qposadr[screw_jid] + 7] = [
            gx, gy, 0.015, 0.7071, 0.0, 0.7071, 0.0
        ]
        mujoco.mj_forward(mj_model, mj_data)
        rend = mujoco.Renderer(mj_model, height=480, width=640)
        cam_map = {
            "cam_high": "overhead_cam",
            "cam_low": "worms_eye_cam",
            "cam_left_wrist": "wrist_cam_left",
            "cam_right_wrist": "wrist_cam_right",
        }
        imgs = {}
        for k, cname in cam_map.items():
            rend.update_scene(mj_data, camera=cname)
            imgs[k] = rend.render().copy()
        rend.close()
        curr_s = np.zeros(14, dtype=np.float32)
        curr_s[0:6] = mj_data.qpos[0:6]
        curr_s[6] = mujoco_to_gripper_rad(mj_data.qpos[6])
        curr_s[7:13] = mj_data.qpos[8:14]
        curr_s[13] = mujoco_to_gripper_rad(mj_data.qpos[14])
        return imgs, curr_s

    mj_raw = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_raw_data = mujoco.MjData(mj_raw)
    cams_raw, s_raw = render_mj_4cam(mj_raw, mj_raw_data, gt_states[0])

    # 3A: Old Bug (uint8 directly)
    pre_bug, norm_bug, phys_bug = run_chunk(cams_raw, s_raw, scale_to_01=False)
    frac_white = float((pre_bug[0] >= 0.999).float().mean())
    l1_bug = float(np.mean(np.abs(phys_bug[:50, :14] - gt_actions[:50])))
    print(f"  [OLD BUG uint8]     _preprocess_images[0]: min={pre_bug[0].min():.2f}, max={pre_bug[0].max():.2f}, frac==+1.0 (WHITE)={frac_white*100:.2f}% | 50-Step L1={l1_bug:.4f}")
    print(f"                      Pred[0]: {np.round(phys_bug[0, :14], 3)}")

    # 3B: Fixed float32 [0,1] on raw XML
    pre_fix, norm_fix, phys_fix = run_chunk(cams_raw, s_raw, scale_to_01=True)
    l1_fix = float(np.mean(np.abs(phys_fix[:50, :14] - gt_actions[:50])))
    print(f"  [FIXED float/255]   _preprocess_images[0]: min={pre_fix[0].min():.2f}, max={pre_fix[0].max():.2f}, mean={pre_fix[0].mean():.2f} | 50-Step L1={l1_fix:.4f}")
    print(f"                      Pred[0]: {np.round(phys_fix[0, :14], 3)}")

    # 3C: Fixed float32 [0,1] + Visual Alignment (FOV + cam_low + white grippers + studio bg)
    mj_aln = mujoco.MjModel.from_xml_path(SCENE_XML)
    apply_visual_alignment(mj_aln)
    mj_aln_data = mujoco.MjData(mj_aln)
    cams_aln, s_aln = render_mj_4cam(mj_aln, mj_aln_data, gt_states[0])
    pre_aln, norm_aln, phys_aln = run_chunk(cams_aln, s_aln, scale_to_01=True)
    l1_aln = float(np.mean(np.abs(phys_aln[:50, :14] - gt_actions[:50])))
    print(f"  [FIXED + VIS ALIGN] _preprocess_images[0]: min={pre_aln[0].min():.2f}, max={pre_aln[0].max():.2f}, mean={pre_aln[0].mean():.2f} | 50-Step L1={l1_aln:.4f}")
    print(f"                      Pred[0]: {np.round(phys_aln[0, :14], 3)}")
    print(f"                      GT[0]  : {np.round(gt_actions[0], 3)}")

    # ------------------------------------------------------------------
    # TEST 4: Verify select_action vs predict_action_chunk equivalence
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("📋 TEST 4: select_action vs predict_action_chunk API Check")
    print("-" * 78)
    policy.reset()
    obs_test = {f"observation.images.{k}": torch.from_numpy(v).permute(2, 0, 1).unsqueeze(0).float() / 255.0 for k, v in cams_aln.items()}
    obs_test["observation.state"] = torch.from_numpy(s_aln).unsqueeze(0)
    obs_test["task"] = [args.task]
    proc_test = preprocess(obs_test)
    with torch.no_grad():
        sa_norm = policy.select_action(proc_test)
    sa_phys = postprocess(sa_norm)
    print(f"  select_action raw output (normalized): shape={tuple(sa_norm.shape)}, min={sa_norm.min():.3f}, max={sa_norm.max():.3f}")
    print(f"  after postprocess (physical rad):      shape={tuple(sa_phys.shape)}, min={sa_phys.min():.3f}, max={sa_phys.max():.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
