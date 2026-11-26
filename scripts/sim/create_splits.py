#!/usr/bin/env python3
"""
Split Manager for Physical AI Dataset
=====================================
Generates configs/dataset_splits.json with the 30 Train / 5 Validation / 15 Test splits.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from physical_ai.data import load_aloha_dataset
from physical_ai.paths import DEFAULT_HF_REPO, SPLITS_JSON


def main():
    print("=" * 65)
    print("📊 CONFIGURING DATASET SPLITS (30 TRAIN / 5 VAL / 15 TEST)")
    print("=" * 65)

    df, _ = load_aloha_dataset(need_videos=False)
    total_episodes = len(df["episode_index"].unique())
    total_frames = len(df)

    train_episodes = list(range(0, 30))
    val_episodes = list(range(30, 35))
    test_episodes = list(range(35, 50))

    train_frames = len(df[df["episode_index"].isin(train_episodes)])
    val_frames = len(df[df["episode_index"].isin(val_episodes)])
    test_frames = len(df[df["episode_index"].isin(test_episodes)])

    splits_info = {
        "dataset": DEFAULT_HF_REPO,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": 50,
        "splits": {
            "train": {
                "episodes": train_episodes,
                "num_episodes": len(train_episodes),
                "num_frames": train_frames,
                "percentage": round(len(train_episodes) / total_episodes * 100, 1),
                "description": "Used for fine-tuning & synthetic generative video augmentations.",
            },
            "validation": {
                "episodes": val_episodes,
                "num_episodes": len(val_episodes),
                "num_frames": val_frames,
                "percentage": round(len(val_episodes) / total_episodes * 100, 1),
                "description": "Used during training to monitor loss & pick the best checkpoint.",
            },
            "test": {
                "episodes": test_episodes,
                "num_episodes": len(test_episodes),
                "num_frames": test_frames,
                "percentage": round(len(test_episodes) / total_episodes * 100, 1),
                "description": "Locked holdout benchmark test set for final evaluation.",
            },
        },
    }

    with open(SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2)

    print("✅ Splits configured successfully:")
    print(f"   • Train Split      : {len(train_episodes)} episodes ({train_frames:,} frames) -> Episodes 0 to 29")
    print(f"   • Validation Split : {len(val_episodes)} episodes ({val_frames:,} frames) -> Episodes 30 to 34")
    print(f"   • Test Split (Eval): {len(test_episodes)} episodes ({test_frames:,} frames) -> Episodes 35 to 49")
    print(f"\n📂 Metadata saved to: {SPLITS_JSON}")


if __name__ == "__main__":
    main()
