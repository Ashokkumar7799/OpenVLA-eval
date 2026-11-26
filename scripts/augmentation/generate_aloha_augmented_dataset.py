#!/usr/bin/env python3
"""
ALOHA Action-Preserving Generative Dataset Augmentation Engine
=============================================================
Transforms the 30 real physical training demonstrations (Episodes 0–29)
into visual variations across 5 color & lighting profiles while preserving
object geometry and 1:1 action correspondence.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from physical_ai.paths import resolve_default_augmented_dir, resolve_local_dataset_dir

DEFAULT_DATASET_DIR = os.path.join(resolve_local_dataset_dir(), "videos")
DEFAULT_OUT_DIR = resolve_default_augmented_dir()

PROFILES = [
    {
        "name": "red_both",
        "desc": "Red Screwdriver + Red Cup (Direct OOD Invariance)",
        "screw_target": {"h": 0, "s_scale": 1.4, "v_scale": 0.85, "fixed_h": True},
        "cup_target": {"h_shift": -98, "s_scale": 1.2, "fixed_h": False},
        "lighting": {"gamma": 1.05, "temp_shift": 5},
    },
    {
        "name": "green_white",
        "desc": "Forest Green Screwdriver + White/Silver Cup",
        "screw_target": {"h": 55, "s_scale": 1.3, "v_scale": 0.75, "fixed_h": True},
        "cup_target": {"desaturate": True, "s_scale": 0.15, "v_scale": 1.4},
        "lighting": {"gamma": 0.95, "temp_shift": -8},
    },
    {
        "name": "orange_gray",
        "desc": "Deep Orange Screwdriver + Dark Slate Gray Cup",
        "screw_target": {"h": 10, "s_scale": 1.5, "v_scale": 0.95, "fixed_h": True},
        "cup_target": {"desaturate": True, "s_scale": 0.1, "v_scale": 0.65},
        "lighting": {"gamma": 1.0, "temp_shift": 10},
    },
    {
        "name": "charcoal_yellow",
        "desc": "Matte Charcoal Screwdriver + Bright Yellow Cup",
        "screw_target": {"desaturate": True, "s_scale": 0.05, "v_scale": 0.45},
        "cup_target": {"h": 26, "s_scale": 1.3, "v_scale": 1.1, "fixed_h": True},
        "lighting": {"gamma": 1.1, "temp_shift": -4},
    },
    {
        "name": "purple_cyan",
        "desc": "Electric Purple Screwdriver + Vivid Cyan Cup",
        "screw_target": {"h": 145, "s_scale": 1.4, "v_scale": 0.85, "fixed_h": True},
        "cup_target": {"h": 90, "s_scale": 1.3, "v_scale": 1.15, "fixed_h": True},
        "lighting": {"gamma": 0.98, "temp_shift": 2},
    },
]


def apply_augmentation(img_bgr, profile):
    """Applies semantic recoloring and photometric transforms while preserving geometry."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b_ch = img_bgr[:, :, 0].astype(np.float32)
    g_ch = img_bgr[:, :, 1].astype(np.float32)
    r_ch = img_bgr[:, :, 2].astype(np.float32)

    yellow_mask = (h >= 16) & (h <= 42) & (s >= 14) & (v >= 95) & (r_ch > b_ch + 12) & (g_ch > b_ch + 12)
    blue_mask = (h >= 85) & (h <= 125) & (s >= 45) & (v >= 50)

    st = profile["screw_target"]
    if st.get("desaturate", False):
        hsv[:, :, 1] = np.where(yellow_mask, np.clip(s * st["s_scale"], 0, 255), hsv[:, :, 1])
        hsv[:, :, 2] = np.where(yellow_mask, np.clip(v * st["v_scale"], 0, 255), hsv[:, :, 2])
    elif st.get("fixed_h", False):
        hsv[:, :, 0] = np.where(yellow_mask, st["h"], hsv[:, :, 0])
        hsv[:, :, 1] = np.where(yellow_mask, np.clip(np.maximum(s * st["s_scale"], s * 2.2 + 80), 0, 245), hsv[:, :, 1])
        hsv[:, :, 2] = np.where(yellow_mask, np.clip(v * st["v_scale"], 0, 240), hsv[:, :, 2])

    ct = profile["cup_target"]
    if ct.get("desaturate", False):
        hsv[:, :, 1] = np.where(blue_mask, np.clip(s * ct["s_scale"], 0, 255), hsv[:, :, 1])
        hsv[:, :, 2] = np.where(blue_mask, np.clip(v * ct.get("v_scale", 1.0), 0, 255), hsv[:, :, 2])
    elif ct.get("fixed_h", False):
        hsv[:, :, 0] = np.where(blue_mask, ct["h"], hsv[:, :, 0])
        hsv[:, :, 1] = np.where(blue_mask, np.clip(s * ct["s_scale"], 0, 255), hsv[:, :, 1])
        hsv[:, :, 2] = np.where(blue_mask, np.clip(v * ct.get("v_scale", 1.0), 0, 255), hsv[:, :, 2])
    elif "h_shift" in ct:
        hsv[:, :, 0] = np.where(blue_mask, (h + ct["h_shift"]) % 180, hsv[:, :, 0])
        hsv[:, :, 1] = np.where(blue_mask, np.clip(s * ct["s_scale"], 0, 255), hsv[:, :, 1])

    result_bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    lit = profile.get("lighting", {})
    gamma = lit.get("gamma", 1.0)
    temp = lit.get("temp_shift", 0)
    if gamma != 1.0 or temp != 0:
        res_f = result_bgr.astype(np.float32)
        if gamma != 1.0:
            res_f = 255.0 * ((res_f / 255.0) ** (1.0 / gamma))
        if temp > 0:
            res_f[:, :, 2] = np.clip(res_f[:, :, 2] + temp * 1.2, 0, 255)
            res_f[:, :, 0] = np.clip(res_f[:, :, 0] - temp * 0.8, 0, 255)
        elif temp < 0:
            res_f[:, :, 0] = np.clip(res_f[:, :, 0] - temp * 1.2, 0, 255)
            res_f[:, :, 2] = np.clip(res_f[:, :, 2] + temp * 0.8, 0, 255)
        result_bgr = np.clip(res_f, 0, 255).astype(np.uint8)

    return result_bgr


def generate_augmented_dataset(args):
    cams = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    video_caps = {}

    print(f"🎬 Initializing camera streams from: {args.dataset_dir}")
    for c in cams:
        p = os.path.join(args.dataset_dir, f"observation.images.{c}", "chunk-000", "file-000.mp4")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing video file for {c} at {p}")
        video_caps[c] = cv2.VideoCapture(p)

    out_clips_dir = os.path.join(args.output_dir, "augmented_clips")
    out_cache_dir = os.path.join(args.output_dir, "dataset_cache_augmented")
    preview_dir = os.path.join(args.output_dir, "preview_grids")
    os.makedirs(out_clips_dir, exist_ok=True)
    os.makedirs(out_cache_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    manifest = []

    print(f"\n🚀 Generating AI Augmented Dataset across {len(args.episodes)} episodes...")
    print(f"🎨 Number of distinct visual profiles: {len(PROFILES)}")

    for idx, ep_id in enumerate(tqdm(args.episodes, desc="Generating Augmented Demonstrations")):
        profile_idx = idx % len(PROFILES)
        profile = PROFILES[profile_idx]

        start_frame = ep_id * 400
        for c in cams:
            video_caps[c].set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out_clip_path = os.path.join(out_clips_dir, f"aug_ep_{ep_id:03d}_{profile['name']}.mp4")
        writer = cv2.VideoWriter(out_clip_path, fourcc, 50.0, (640, 480))
        preview_saved = False

        for f_offset in range(400):
            abs_idx = start_frame + f_offset
            tiles = []
            for c in cams:
                ret, frame = video_caps[c].read()
                if not ret:
                    break
                tiles.append(cv2.resize(frame, (320, 240)))

            if len(tiles) < 4:
                break

            raw_mosaic = np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])
            aug_mosaic = apply_augmentation(raw_mosaic, profile)
            writer.write(aug_mosaic)

            cache_frame_path = os.path.join(out_cache_dir, f"frame_aug_{abs_idx:06d}.jpg")
            cv2.imwrite(cache_frame_path, aug_mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if f_offset == 0 and not preview_saved:
                prev_path = os.path.join(preview_dir, f"preview_profile_{profile_idx}_{profile['name']}.jpg")
                cv2.imwrite(prev_path, np.hstack([raw_mosaic, aug_mosaic]))
                preview_saved = True

        writer.release()

        manifest.append({
            "augmented_episode_id": ep_id + 100,
            "source_episode_id": ep_id,
            "profile_name": profile["name"],
            "profile_desc": profile["desc"],
            "clip_path": os.path.relpath(out_clip_path, args.output_dir),
            "start_frame_abs": start_frame,
            "total_frames": 400,
        })

    for c in cams:
        video_caps[c].release()

    manifest_path = os.path.join(args.output_dir, "augmented_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Augmented Dataset Generation Complete!")
    print(f"📦 Total Augmented Clips: {len(manifest)} saved in {out_clips_dir}")
    print(f"🖼️ Pre-extracted Training Cache: {out_cache_dir}")
    print(f"📄 Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ALOHA Augmented Dataset")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    return parser.parse_args()


if __name__ == "__main__":
    generate_augmented_dataset(parse_args())
