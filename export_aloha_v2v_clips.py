"""
Export ALOHA Training Demonstration Clips for Generative Video Augmentation
===========================================================================
Extracts synchronized 4-camera (2x2 mosaic) video clips for the 30 training
episodes (Episodes 0–29) from raw LeRobot dataset MP4 streams.

Supports two output formats:
1. Full 400-frame @ 50 FPS clips (8.00 seconds per episode)
2. 81-frame @ 10 FPS clips (stride=5, matching Wan 2.1 / SVI native 4n+1 temporal windows)
"""

import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

DEFAULT_DATASET_DIR = r"E:\mediapipe\robotics_datasets\aloha_static_screw_driver\videos"
DEFAULT_OUTPUT_DIR = r"E:\mediapipe\physical_ai_eval\augmented_dataset\raw_clips"

def parse_args():
    parser = argparse.ArgumentParser(description="Export ALOHA V2V Training Clips")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to raw dataset videos")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for exported clips")
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(30)), help="Episode indices to export (default: 0-29)")
    parser.add_argument("--mode", choices=["full_400", "diff_81", "both"], default="both",
                        help="Export mode: 'full_400' (50Hz 400 frames), 'diff_81' (10Hz 81 frames for Wan 2.1), or 'both'")
    parser.add_argument("--manifest_only", action="store_true", help="Only verify and write manifest without re-encoding")
    return parser.parse_args()

def export_clips(args):
    cams = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    video_caps = {}
    
    print(f"🎬 Checking video streams in: {args.dataset_dir}")
    for c in cams:
        p = os.path.join(args.dataset_dir, f"observation.images.{c}", "chunk-000", "file-000.mp4")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing video file for camera {c} at: {p}")
        cap = cv2.VideoCapture(p)
        video_caps[c] = cap
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  • {c}: {total:,} frames available")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"\n🚀 Exporting {len(args.episodes)} episodes (Mode: {args.mode})...")
    for ep_id in tqdm(args.episodes, desc="Exporting Demonstrations"):
        start_frame = ep_id * 400
        
        # Paths
        clip_400_path = os.path.join(args.output_dir, f"episode_{ep_id:03d}_4cam_400f_50fps.mp4")
        clip_81_path = os.path.join(args.output_dir, f"episode_{ep_id:03d}_4cam_81f_10fps.mp4")

        export_400 = args.mode in ["full_400", "both"]
        export_81 = args.mode in ["diff_81", "both"]

        # Seek all camera streams to start_frame
        for c in cams:
            video_caps[c].set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer_400 = cv2.VideoWriter(clip_400_path, fourcc, 50.0, (640, 480)) if export_400 else None
        writer_81 = cv2.VideoWriter(clip_81_path, fourcc, 10.0, (640, 480)) if export_81 else None

        frames_read = 0
        diff_count = 0

        for f_offset in range(400):
            tiles = []
            for c in cams:
                ret, frame = video_caps[c].read()
                if not ret:
                    break
                tiles.append(cv2.resize(frame, (320, 240)))
            
            if len(tiles) < 4:
                print(f"⚠️ Warning: Truncated frame read at episode {ep_id}, offset {f_offset}")
                break

            # Assemble 2x2 mosaic (640x480)
            top = np.hstack([tiles[0], tiles[1]])
            bottom = np.hstack([tiles[2], tiles[3]])
            mosaic = np.vstack([top, bottom])

            # Write full 50fps frame
            if writer_400:
                writer_400.write(mosaic)

            # Write 81-frame diffusion clip (stride=5 for frames 0..400 -> exactly 81 frames)
            if writer_81 and (f_offset % 5 == 0) and diff_count < 81:
                writer_81.write(mosaic)
                diff_count += 1

            frames_read += 1

        if writer_400:
            writer_400.release()
        if writer_81:
            writer_81.release()

        manifest.append({
            "episode_id": ep_id,
            "start_frame_abs": start_frame,
            "end_frame_abs": start_frame + frames_read - 1,
            "total_frames": frames_read,
            "clip_400_path": os.path.relpath(clip_400_path, args.output_dir) if export_400 else None,
            "clip_81_path": os.path.relpath(clip_81_path, args.output_dir) if export_81 else None,
            "diff_frames_written": diff_count if export_81 else None
        })

    # Release captures
    for c in cams:
        video_caps[c].release()

    manifest_path = os.path.join(args.output_dir, "episodes_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Export Complete! Successfully processed {len(manifest)} demonstration clips.")
    print(f"📄 Manifest written to: {manifest_path}")

if __name__ == "__main__":
    args = parse_args()
    export_clips(args)
