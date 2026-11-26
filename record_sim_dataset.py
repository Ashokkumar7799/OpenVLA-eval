#!/usr/bin/env python3
"""
ALOHA MuJoCo Digital Twin Dataset Recorder
==========================================
Replays Ground-Truth trajectories inside the calibrated MuJoCo Digital Twin
at 50 Hz (10 substeps * 0.002s) and records a synchronized 4-camera LeRobot
simulation dataset (MP4 camera streams + Parquet states/actions + previews).

Reuses camera/gripper calibration helpers directly from pi05_runner.py.
"""

import os
import sys
if sys.platform != "win32" and "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pi05_runner import (
    LOCAL_DATASET_DIR,
    SCENE_XML,
    apply_visual_alignment,
    calibrate_screwdriver_contact,
    compute_episode_grasp_point,
    map_action_to_ctrl,
    mujoco_to_gripper_rad,
    scale_gripper,
)

CAM_MAP = {
    "cam_high": "overhead_cam",
    "cam_low": "worms_eye_cam",
    "cam_left_wrist": "wrist_cam_left",
    "cam_right_wrist": "wrist_cam_right",
}
DEFAULT_TASK = "Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup."


def parse_args():
    parser = argparse.ArgumentParser(description="Record MuJoCo Digital Twin LeRobot Dataset")
    parser.add_argument("--episodes", type=int, nargs="+", default=[1],
                        help="Source trajectory episode IDs to record in MuJoCo (default: [1])")
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "sim_aloha_dataset"),
                        help="Output directory for generated dataset & previews")
    parser.add_argument("--shadow_size", type=int, default=1024,
                        help="MuJoCo shadow map resolution (default: 1024)")
    parser.add_argument("--no_visual_align", action="store_true",
                        help="Disable camera FOV / cam_low / white-gripper visual alignment")
    parser.add_argument("--save_preview", action="store_true",
                        help="Also save 2x2 4-cam MP4 and Real-vs-Sim side-by-side preview")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional HF token if dataset not local")
    return parser.parse_args()


def load_source_dataset(hf_token=None):
    """Loads source dataset parquets (preferring local dataset when present)."""
    if os.path.exists(LOCAL_DATASET_DIR):
        repo_dir = LOCAL_DATASET_DIR
    else:
        from huggingface_hub import snapshot_download
        token = hf_token or os.environ.get("HF_TOKEN")
        repo_dir = snapshot_download(
            "lerobot/aloha_static_screw_driver",
            repo_type="dataset",
            allow_patterns=["*.parquet", "meta/**", "videos/**"],
            token=token,
        )
    parquet_files = sorted(
        p for p in glob.glob(os.path.join(repo_dir, "**", "*.parquet"), recursive=True)
        if "meta" not in p
    )
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    return df, repo_dir


def open_real_video_caps(repo_dir):
    """Opens cv2.VideoCapture handles for real 4-camera comparison previews."""
    caps = {}
    for cam_key in CAM_MAP:
        matches = sorted(glob.glob(os.path.join(repo_dir, "videos", f"*{cam_key}*", "**", "*.mp4"), recursive=True))
        if matches:
            caps[cam_key] = cv2.VideoCapture(matches[0])
    return caps


def make_mosaic(cam_frames_rgb, tile_w=320, tile_h=240):
    """Builds 2x2 RGB mosaic ordered [cam_high, cam_low; cam_left_wrist, cam_right_wrist]."""
    tiles = [
        cv2.resize(cam_frames_rgb[k], (tile_w, tile_h))
        for k in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
    ]
    return np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])


def write_video_mp4(path, frames_rgb, fps=50.0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import mediapy as media
        media.write_video(path, frames_rgb, fps=int(fps))
    except Exception:
        h, w = frames_rgb[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        for fr in frames_rgb:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()


def main():
    args = parse_args()
    import mujoco

    os.makedirs(args.output_dir, exist_ok=True)
    df, repo_dir = load_source_dataset(args.hf_token)

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    calibrate_screwdriver_contact(model)
    if not args.no_visual_align:
        apply_visual_alignment(model, shadow_size=args.shadow_size)
    else:
        model.vis.quality.shadowsize = int(args.shadow_size)

    screw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
    screw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")

    real_caps = open_real_video_caps(repo_dir) if args.save_preview else {}
    all_rows = []
    cam_buffers = {k: [] for k in CAM_MAP}
    episode_summaries = []
    global_index = 0

    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        for new_ep_idx, src_ep_id in enumerate(args.episodes):
            ep_df = df[df["episode_index"] == src_ep_id].sort_values("frame_index")
            if len(ep_df) == 0:
                print(f"⚠️ Skipping missing episode {src_ep_id}")
                continue

            gt_actions = np.array([r["action"] for _, r in ep_df.iterrows()], dtype=np.float32)
            gt_states = (
                np.array([r["observation.state"] for _, r in ep_df.iterrows()], dtype=np.float32)
                if "observation.state" in ep_df.columns
                else gt_actions
            )
            src_abs_start = int(ep_df["index"].iloc[0])
            gx, gy = compute_episode_grasp_point(model, data, gt_actions)

            mujoco.mj_resetData(model, data)
            init = gt_states[0]
            data.qpos[0:6] = init[0:6]
            data.qpos[6] = data.qpos[7] = scale_gripper(init[6])
            data.qpos[8:14] = init[7:13]
            data.qpos[14] = data.qpos[15] = scale_gripper(init[13])
            data.ctrl[:14] = map_action_to_ctrl(init)
            if screw_jid != -1:
                qadr = model.jnt_qposadr[screw_jid]
                data.qpos[qadr:qadr + 7] = [gx, gy, 0.015, 0.7071, 0.0, 0.7071, 0.0]
            mujoco.mj_forward(model, data)

            init_z = float(data.xpos[screw_bid][2]) if screw_bid != -1 else 0.0
            max_z = init_z
            sim_mosaics = []
            real_mosaics = []
            keyframe_strips = []
            keyframe_indices = {0, 100, 175, 250, 350}

            if real_caps:
                for cap in real_caps.values():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_abs_start)

            t0 = time.time()
            n_frames = len(gt_actions)
            pbar = tqdm(range(n_frames), desc=f"🎬 Recording Sim Ep {src_ep_id}", unit="frame")
            for f_idx in pbar:
                # 1. Record observation BEFORE applying action[f_idx] (standard MDP transition)
                frame_cams = {}
                for cam_key, mj_cam in CAM_MAP.items():
                    renderer.update_scene(data, camera=mj_cam)
                    rgb = renderer.render().copy()
                    frame_cams[cam_key] = rgb
                    cam_buffers[cam_key].append(rgb)

                sim_state = np.zeros(14, dtype=np.float32)
                sim_state[0:6] = data.qpos[0:6]
                sim_state[6] = mujoco_to_gripper_rad(data.qpos[6])
                sim_state[7:13] = data.qpos[8:14]
                sim_state[13] = mujoco_to_gripper_rad(data.qpos[14])

                act = gt_actions[f_idx].astype(np.float32)
                all_rows.append({
                    "observation.state": sim_state.tolist(),
                    "action": act.tolist(),
                    "episode_index": int(new_ep_idx),
                    "source_episode_index": int(src_ep_id),
                    "frame_index": int(f_idx),
                    "timestamp": np.float32(f_idx * 0.02),
                    "next.done": bool(f_idx == n_frames - 1),
                    "index": int(global_index),
                    "task_index": 0,
                })
                global_index += 1

                if args.save_preview:
                    sim_mos = make_mosaic(frame_cams)
                    sim_mosaics.append(sim_mos)
                    if len(real_caps) == len(CAM_MAP):
                        real_cams = {}
                        ok = True
                        for cam_key, cap in real_caps.items():
                            ret, bgr = cap.read()
                            if not ret or bgr is None:
                                ok = False
                                break
                            real_cams[cam_key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if ok:
                            real_mos = make_mosaic(real_cams)
                            real_mosaics.append(real_mos)
                            if f_idx in keyframe_indices:
                                pair = np.hstack([real_mos, sim_mos])
                                keyframe_strips.append((f_idx, pair))

                # 2. Advance MuJoCo 50 Hz physics (10 substeps * 0.002s = 20ms)
                data.ctrl[:14] = map_action_to_ctrl(act)
                for _ in range(10):
                    mujoco.mj_step(model, data)

                if screw_bid != -1:
                    max_z = max(max_z, float(data.xpos[screw_bid][2]))

            elapsed = time.time() - t0
            lift_cm = (max_z - init_z) * 100.0
            final_pos = data.xpos[screw_bid] if screw_bid != -1 else np.zeros(3)
            cup_pos = model.body_pos[cup_bid] if cup_bid != -1 else np.array([-0.12, 0.19, 0.001])
            cup_dist_cm = float(np.linalg.norm(final_pos[:2] - cup_pos[:2]) * 100.0)
            cup_success = bool(cup_dist_cm < 10.0 and final_pos[2] < 0.19)

            summary = {
                "sim_episode_index": int(new_ep_idx),
                "source_episode_index": int(src_ep_id),
                "num_frames": int(n_frames),
                "grasp_origin_xy": [round(gx, 4), round(gy, 4)],
                "lift_height_cm": round(float(lift_cm), 2),
                "lift_success": bool(lift_cm >= 5.0),
                "cup_dist_cm": round(cup_dist_cm, 2),
                "cup_success": cup_success,
                "record_time_sec": round(elapsed, 2),
            }
            episode_summaries.append(summary)
            print(f"✅ Ep {src_ep_id}: Lift={lift_cm:.2f}cm ({summary['lift_success']}) | Cup={cup_dist_cm:.2f}cm ({cup_success}) | {elapsed:.1f}s")

            if args.save_preview:
                prev_dir = os.path.join(args.output_dir, "previews")
                os.makedirs(prev_dir, exist_ok=True)
                sim_prev_path = os.path.join(prev_dir, f"ep{src_ep_id}_sim_4cam.mp4")
                write_video_mp4(sim_prev_path, sim_mosaics, fps=50.0)
                print(f"🎥 Saved Sim 4-cam preview -> {sim_prev_path}")
                if real_mosaics:
                    comb = [np.hstack([r, s]) for r, s in zip(real_mosaics, sim_mosaics)]
                    comb_path = os.path.join(prev_dir, f"ep{src_ep_id}_real_vs_sim_4cam.mp4")
                    write_video_mp4(comb_path, comb, fps=50.0)
                    print(f"🎥 Saved Real-vs-Sim 4-cam comparison -> {comb_path}")
                if keyframe_strips:
                    sheet = np.vstack([pair for _, pair in keyframe_strips])
                    sheet_path = os.path.join(prev_dir, f"ep{src_ep_id}_keyframes_real_vs_sim.jpg")
                    cv2.imwrite(sheet_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    print(f"🖼️ Saved Keyframe comparison sheet -> {sheet_path}")
    finally:
        renderer.close()
        for cap in real_caps.values():
            cap.release()

    # Save LeRobot-style Parquet + Videos + Metadata
    data_out = os.path.join(args.output_dir, "data", "chunk-000", "file-000.parquet")
    os.makedirs(os.path.dirname(data_out), exist_ok=True)
    pd.DataFrame(all_rows).to_parquet(data_out, index=False)

    for cam_key, frames_list in cam_buffers.items():
        v_out = os.path.join(args.output_dir, "videos", f"observation.images.{cam_key}", "chunk-000", "file-000.mp4")
        write_video_mp4(v_out, frames_list, fps=50.0)

    meta_dir = os.path.join(args.output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[DEFAULT_TASK])
    tasks_df.to_parquet(os.path.join(meta_dir, "tasks.parquet"))

    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "total_episodes": len(episode_summaries),
        "total_frames": len(all_rows),
        "total_tasks": 1,
        "fps": 50,
        "visual_align": bool(not args.no_visual_align),
        "episode_summaries": episode_summaries,
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n📦 Simulation dataset saved to: {args.output_dir}")
    print(f"   • Parquet rows : {len(all_rows)} -> {data_out}")
    print(f"   • Camera MP4s  : {len(cam_buffers)} streams in {os.path.join(args.output_dir, 'videos')}")


if __name__ == "__main__":
    main()
