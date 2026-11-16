"""
Split Manager for Physical AI Dataset
Formalizes the 30 Train / 5 Validation / 15 Test Evaluation Split.
"""

import json
import pandas as pd

PARQUET_FILE = r"E:\mediapipe\robotics_datasets\aloha_static_screw_driver\data\chunk-000\file-000.parquet"
OUTPUT_SPLIT_JSON = r"E:\mediapipe\physical_ai_eval\dataset_splits.json"

print("=" * 65)
print("📊 CONFIGURING DATASET SPLITS (30 TRAIN / 5 VAL / 15 TEST)")
print("=" * 65)

df = pd.read_parquet(PARQUET_FILE)
total_episodes = len(df["episode_index"].unique())
total_frames = len(df)

# Define Splits:
# Train: 30 episodes (0..29) -> 60%
# Validation: 5 episodes (30..34) -> 10%
# Test: 15 episodes (35..49) -> 30% (Heavy focus on rigorous testing)
train_episodes = list(range(0, 30))
val_episodes = list(range(30, 35))
test_episodes = list(range(35, 50))

train_frames = len(df[df["episode_index"].isin(train_episodes)])
val_frames = len(df[df["episode_index"].isin(val_episodes)])
test_frames = len(df[df["episode_index"].isin(test_episodes)])

splits_info = {
    "dataset": "lerobot/aloha_static_screw_driver",
    "total_episodes": total_episodes,
    "total_frames": total_frames,
    "fps": 50,
    "splits": {
        "train": {
            "episodes": train_episodes,
            "num_episodes": len(train_episodes),
            "num_frames": train_frames,
            "percentage": round(len(train_episodes) / total_episodes * 100, 1),
            "description": "Used for OpenVLA LoRA fine-tuning & synthetic video augmentations."
        },
        "validation": {
            "episodes": val_episodes,
            "num_episodes": len(val_episodes),
            "num_frames": val_frames,
            "percentage": round(len(val_episodes) / total_episodes * 100, 1),
            "description": "Used during training to monitor loss & pick the best checkpoint."
        },
        "test": {
            "episodes": test_episodes,
            "num_episodes": len(test_episodes),
            "num_frames": test_frames,
            "percentage": round(len(test_episodes) / total_episodes * 100, 1),
            "description": "Locked holdout benchmark test set for final scientific evaluation."
        }
    }
}

with open(OUTPUT_SPLIT_JSON, "w") as f:
    json.dump(splits_info, f, indent=2)

print(f"✅ Splits configured successfully:")
print(f"   • Train Split      : {len(train_episodes)} episodes ({train_frames:,} frames) -> Episodes 0 to 29")
print(f"   • Validation Split : {len(val_episodes)} episodes ({val_frames:,} frames) -> Episodes 30 to 34")
print(f"   • Test Split (Eval): {len(test_episodes)} episodes ({test_frames:,} frames) -> Episodes 35 to 49")
print(f"\n📂 Metadata saved to: {OUTPUT_SPLIT_JSON}")
print("=" * 65)

