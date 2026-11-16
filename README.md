# 🤖 Physical AI: Bimanual ALOHA Digital Twin & OpenVLA Evaluation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%204.0.1-orange.svg)](https://mujoco.org/)
[![OpenVLA](https://img.shields.io/badge/VLA-OpenVLA--7B-purple.svg)](https://github.com/openvla/openvla)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A production-grade **Physical AI Evaluation and Simulation Layer** for bimanual robotic manipulation, built around the canonical `lerobot/aloha_static_screw_driver` dataset.

This repository provides:
1. **MuJoCo 1-to-1 Digital Twin**: Exact physics model of the DeepMind ALOHA bimanual workspace (matte black table, white reference line, blue cup with yellow foam insert, and 6-DoF yellow screwdriver).
2. **Benchmark Engine**: Automated 15-episode evaluation harness computing **Task Success Rate** ($\ge 2\text{ cm}$ physical object lift) and joint-space L1 errors across 14 degrees of freedom.
3. **OpenVLA-7B Colab Integration**: Self-contained GPU notebook for zero-shot evaluation and LoRA parameter-efficient fine-tuning on 30 training episodes.

---

## 📊 Dataset Splits & Benchmarking Philosophy

| Split | Episodes | Frames (50 FPS) | Purpose |
| :--- | :---: | :---: | :--- |
| **Train** | 0 – 29 (30 eps) | 12,000 | LoRA fine-tuning & synthetic domain augmentation |
| **Validation** | 30 – 34 (5 eps) | 2,000 | Hyperparameter tuning & checkpoint selection |
| **Test (Holdout)** | 35 – 49 (15 eps) | 6,000 | Untouched holdout evaluation benchmark |

### What is "Task Success Rate"?
In physical robotics, low neural MSE does not guarantee task completion. Our MuJoCo engine verifies physical interaction:
$$\text{Task Success} = \begin{cases} \text{True} & \text{if screwdriver is physically grasped and lifted } \ge 2.0\text{ cm} \\ \text{False} & \text{otherwise} \end{cases}$$

---

## 🚀 Quickstart: Local MuJoCo Interactive Simulation

### 1. Installation
```bash
git clone https://github.com/<your-username>/aloha-openvla-eval.git
cd aloha-openvla-eval

python -m venv venv
venv\Scripts\activate  # On Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Interactive 3D Viewer
Run the native 3D MuJoCo desktop viewer to inspect Episode 0 playback with calibrated Dynamixel grippers:
```bash
python live_viewer.py
```
* **Left Click + Drag**: Orbit / Rotate camera
* **Right Click + Drag**: Zoom in / out
* **Middle Click**: Pan camera
* **Spacebar**: Pause / Resume

### 3. Automated 15-Episode Test Evaluation
```bash
python evaluate_baseline.py
```
Outputs `baseline_benchmark_report.json` with per-episode lift metrics and joint-space L1 errors.

---

## ⚡ Google Colab GPU Workflow (OpenVLA-7B)

1. Open `openvla_aloha_eval_and_finetune.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select runtime: **L4 GPU** or **A100 GPU**.
3. Run all cells:
   * Automatically downloads `lerobot/aloha_static_screw_driver`.
   * Loads `openvla/openvla-7b` in 4-bit / bfloat16.
   * Runs zero-shot evaluation on the 15 test episodes.
   * Fine-tunes with LoRA (Rank 32) on the 30 training episodes.
   * Renders in-browser simulation videos of the robot manipulating the screwdriver.

---

## 📁 Repository Structure

```
├── mujoco_menagerie/aloha/
│   ├── scene_with_screwdriver.xml   # Digital twin scene (arms, table, cup, screwdriver)
│   ├── aloha.xml                    # ALOHA 14-DoF kinematic definition
│   └── assets/                      # STL meshes & textures
├── dataset_splits.json              # Canonical 30 Train / 5 Val / 15 Test splits
├── create_splits.py                 # Split generator script
├── live_viewer.py                   # Real-time interactive 3D MuJoCo viewer
├── evaluate_baseline.py             # 15-episode test evaluation engine
├── baseline_benchmark_report.json   # Benchmark scorecard & metrics
├── openvla_aloha_eval_and_finetune.ipynb  # Google Colab GPU notebook
├── colab_runner.py                  # Headless Cloud/Colab runner
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```
