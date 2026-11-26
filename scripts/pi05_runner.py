#!/usr/bin/env python3
"""
Pi0.5 (π₀.₅) LeRobot Fine-Tuning & MuJoCo Evaluation Runner
============================================================
CLI entrypoint for training Pi0.5 with LeRobot and running 50 Hz closed-loop
action-chunked rollouts inside the calibrated ALOHA MuJoCo Digital Twin.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from physical_ai.paths import DEFAULT_HF_REPO
from physical_ai.sim_twin import DEFAULT_TASK, ensure_mujoco_gl

ensure_mujoco_gl()

from physical_ai.pi05_policy import run_pi05_evaluation, run_pi05_training


def parse_args():
    parser = argparse.ArgumentParser(description="Pi0.5 Aloha Runner (Training & MuJoCo Rollout)")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval",
                        help="'train' (launches LeRobot Pi0.5 fine-tuning) or 'eval' (MuJoCo Digital Twin rollout)")
    parser.add_argument("--policy_path", type=str, default="./outputs/pi05_aloha_expert/checkpoints/last/pretrained_model",
                        help="Path to trained Pi0.5 checkpoint or Hugging Face repo (or 'none' for GT replay)")
    parser.add_argument("--dataset_repo", type=str, default=DEFAULT_HF_REPO,
                        help="Hugging Face dataset repository ID for training")
    parser.add_argument("--episodes", type=int, nargs="+", default=[35],
                        help="Holdout test episode IDs to evaluate (e.g. --episodes 35 36 37)")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps for LeRobot train (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for single 24GB L4 GPU (default: 8)")
    parser.add_argument("--exec_steps", type=int, default=25,
                        help="Action steps executed per predicted chunk before replanning (default: 25 = 0.5s closed-loop)")
    parser.add_argument("--no_visual_align", action="store_true",
                        help="Disable Sim-to-Real camera FOV/cam_low/white-gripper visual alignment")
    parser.add_argument("--shadow_size", type=int, default=1024,
                        help="MuJoCo shadow map resolution (1024 is ~2.4x faster than 8192)")
    parser.add_argument("--ood_red", action="store_true", help="OOD Stress Test: Recolors tool & cup to RED")
    parser.add_argument("--save_video", action="store_true", help="Record and save MP4 rollout video")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi05_aloha_expert", help="Output directory")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Natural language task instruction for Pi0.5 VLA policy")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_pi05_training(args)
    else:
        run_pi05_evaluation(args)
