"""Dataset, video I/O, split configuration, and checkpoint resolution helpers."""

import glob
import json
import os
import cv2
import numpy as np
import pandas as pd

from .paths import DEFAULT_HF_REPO, SPLITS_JSON, resolve_local_dataset_dir
from .sim_twin import CAM_MAP


def load_dataset_splits(splits_path: str = SPLITS_JSON) -> dict:
    """Loads the canonical train/val/test split manifest."""
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_aloha_dataset(hf_token=None, need_videos: bool = False, repo_id: str = DEFAULT_HF_REPO):
    """Loads LeRobot parquet telemetry, preferring the local mirror when present."""
    local_dir = resolve_local_dataset_dir()
    if os.path.exists(local_dir):
        repo_dir = local_dir
    else:
        from huggingface_hub import snapshot_download

        token = hf_token or os.environ.get("HF_TOKEN")
        patterns = ["*.parquet", "meta/**"]
        if need_videos:
            patterns.append("videos/**")
        repo_dir = snapshot_download(
            repo_id,
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


def resolve_policy_path(policy_path: str, output_dir: str = "./outputs") -> str:
    """Resolves local checkpoint path, HF repo ID, or auto-detects latest pretrained_model."""
    is_explicit_none = policy_path.lower() in {"none", "gt", "ground_truth"}
    if is_explicit_none or os.path.exists(policy_path):
        return policy_path
    if "/" in policy_path and not policy_path.startswith((".", "/")):
        return policy_path
    candidates = glob.glob(os.path.join(output_dir, "**", "pretrained_model"), recursive=True)
    if candidates:
        candidates.sort()
        return candidates[-1]
    return policy_path


def make_4cam_mosaic(cam_frames_rgb: dict, tile_w: int = 320, tile_h: int = 240) -> np.ndarray:
    """Builds a 2x2 RGB mosaic ordered [cam_high, cam_low; cam_left_wrist, cam_right_wrist]."""
    tiles = [
        cv2.resize(cam_frames_rgb[k], (tile_w, tile_h))
        for k in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
    ]
    return np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])


def write_video_mp4(path: str, frames_rgb: list, fps: float = 50.0):
    """Writes RGB frames to MP4 using mediapy when available, falling back to OpenCV."""
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


def read_real_4cam_frame(repo_dir: str, abs_frame_idx: int):
    """Reads all four synchronized real-camera RGB frames for a global dataset frame index."""
    out = {}
    for cam_key in CAM_MAP:
        matches = sorted(glob.glob(os.path.join(repo_dir, "videos", f"*{cam_key}*", "**", "*.mp4"), recursive=True))
        if not matches:
            return None
        cap = cv2.VideoCapture(matches[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(abs_frame_idx))
        ret, bgr = cap.read()
        cap.release()
        if not ret or bgr is None:
            return None
        out[cam_key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out
