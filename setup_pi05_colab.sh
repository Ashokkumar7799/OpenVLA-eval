#!/bin/bash
# ==============================================================================
# Pi0.5 (π₀.₅) LeRobot Automated Colab Setup Script (NVIDIA L4 / A100 Optimized)
# ==============================================================================

set -e

echo "================================================================="
echo "📦 INSTALLING ALL LEROBOT, PI0.5 & MUJOCO DEPENDENCIES"
echo "================================================================="
pip install -q --upgrade pip
pip install -r requirements.txt

echo "================================================================="
echo "🔍 VERIFYING GPU & ENVIRONMENT"
echo "================================================================="
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
    print('Total VRAM: {:.2f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1e9))
"

echo "================================================================="
echo "🚀 READY TO LAUNCH PI0.5 TRAINING & EVALUATION"
echo "================================================================="
echo "To start Pi0.5 fine-tuning on 30 ALOHA episodes, run:"
echo "python pi05_runner.py --mode train --batch_size 8 --steps 5000"
echo ""
echo "To evaluate in calibrated MuJoCo Digital Twin, run:"
echo "python pi05_runner.py --mode eval --episodes 35 36 37 --save_video"
