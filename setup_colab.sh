#!/bin/bash
set -e

echo "================================================================="
echo "📦 INSTALLING ALL OPENVLA & MUJOCO DEPENDENCIES"
echo "================================================================="
pip install -q -r requirements.txt

echo "================================================================="
echo "🚀 LAUNCHING OPENVLA EVALUATION ON L4 GPU"
echo "================================================================="
python colab_runner.py --mode zero_shot --test_limit 5
