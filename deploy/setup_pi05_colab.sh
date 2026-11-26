#!/bin/bash
# ==============================================================================
# Pi0.5 (π₀.₅) LeRobot Automated Colab / Cloud Setup Script (NVIDIA L4 / A100)
# ==============================================================================

set -e

echo "================================================================="
echo "📦 INSTALLING ALL LEROBOT, PI0.5, MUJOCO & PACKAGE DEPENDENCIES"
echo "================================================================="
pip install -q --upgrade pip
pip install -r requirements.txt
pip install -e .

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
echo "1. Run systematic diagnostics (verify scaling & weights):"
echo "   python scripts/debug_pi05_pipeline.py --episode 35"
echo ""
echo "2. Start Pi0.5 fine-tuning on ALOHA episodes:"
echo "   python scripts/pi05_runner.py --mode train --batch_size 8 --steps 5000"
echo ""
echo "3. Evaluate closed-loop in calibrated MuJoCo Digital Twin:"
echo "   python scripts/pi05_runner.py --mode eval --episodes 35 36 37 --save_video"
