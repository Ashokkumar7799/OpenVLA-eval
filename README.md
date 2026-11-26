# Physical AI Evaluation & Generative Augmentation Engine

A modular evaluation, simulation, and action-preserving augmentation platform for bimanual robotic manipulation on **`lerobot/aloha_static_screw_driver`** (14-DoF ALOHA, 50 Hz, 4 synchronized cameras).

It provides:
1. **Shared Core Package (`src/physical_ai/`)**: Single source of truth for MuJoCo Digital Twin calibration (fingertip grasp-site placement, `0.010` screwdriver torsional friction, `0.0 mm` shaft contact margin, camera FOV/color alignment), LeRobot data loading, and $\pi_{0.5}$ closed-loop inference.
2. **Closed-Loop Digital Twin Evaluation**: Verified at **50/50 Lift** and **47/50 Cup Placement** across all 50 Ground-Truth demonstrations (`35–49` locked holdout split, plus `--ood_red` stress testing).
3. **Action-Preserving Augmentation Pipelines (`scripts/augmentation/`)**: Both fast semantic HSV/photometric transforms and synchronized `81`-frame / `400`-frame V2V video export for diffusion pipelines.

---

## Repository Layout

```text
physical_ai_eval/
├── pyproject.toml                       # Installable package metadata (`pip install -e .`)
├── requirements.txt                     # Active Pi0.5, LeRobot, and MuJoCo dependencies
├── .gitignore
│
├── configs/
│   ├── dataset_splits.json              # Canonical 30 Train (0-29) / 5 Val (30-34) / 15 Test (35-49)
│   └── baseline_benchmark_report.json   # Reference zero-shot vs human benchmark report
│
├── deploy/
│   └── setup_pi05_colab.sh              # One-click Colab / Cloud GPU setup script
│
├── docs/
│   └── PIPELINE_DOCUMENTATION.md        # Detailed mathematical & physics calibration reference
│
├── src/physical_ai/                     # Core reusable library (cross-platform paths, zero hardcoding)
│   ├── __init__.py
│   ├── paths.py                         # Repo, XML, config & ALOHA_DATASET_DIR resolution
│   ├── data.py                          # LeRobot dataset, split & video I/O helpers
│   ├── sim_twin.py                      # Gripper scaling, grasp sites, contact & visual alignment
│   └── pi05_policy.py                   # Pi0.5 training launcher & 50 Hz closed-loop evaluator
│
├── scripts/                             # Runnable CLI entrypoints
│   ├── pi05_runner.py                   # Train / evaluate Pi0.5 closed-loop in MuJoCo
│   ├── debug_pi05_pipeline.py           # 4-stage diagnostics (GT physics, real L1, uint8 vs float32, API)
│   ├── record_sim_dataset.py            # Record 50 Hz 4-camera LeRobot v3.0 datasets in MuJoCo
│   ├── augmentation/
│   │   ├── generate_aloha_augmented_dataset.py  # Semantic color/lighting profile augmentation
│   │   └── export_aloha_v2v_clips.py            # Synchronized 4-cam clip exporter for V2V diffusion
│   └── sim/
│       ├── create_splits.py             # Regenerate configs/dataset_splits.json
│       ├── generate_cup.py              # Procedural tapered cup mesh generator
│       └── live_viewer.py               # Interactive 3D desktop MuJoCo replay viewer
│
├── mujoco_menagerie/                    # DeepMind ALOHA MJCF XML, meshes & scene_with_screwdriver.xml
└── legacy/openvla/                      # Archived OpenVLA-7B v1 notebooks, LoRA training & scripts
```

---

## Quickstart

### 1. Local Setup (Windows / Linux)
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
Optional environment overrides for custom storage locations:
```bash
export ALOHA_DATASET_DIR=/path/to/aloha_static_screw_driver
export ALOHA_AUGMENTED_DIR=/path/to/augmented_dataset
```

### 2. Cloud / Google Colab (GPU)
```bash
git clone https://github.com/Ashokkumar7799/OpenVLA-eval.git
cd OpenVLA-eval
bash deploy/setup_pi05_colab.sh
```

---

## Key Commands

- **Verify Physics & Image Scaling Diagnostics (4 Tests):**
  ```bash
  python scripts/debug_pi05_pipeline.py --episode 35
  ```
- **Evaluate $\pi_{0.5}$ Closed-Loop in MuJoCo (Standard & OOD Red):**
  ```bash
  python scripts/pi05_runner.py --mode eval --episodes 35 36 37 --save_video
  python scripts/pi05_runner.py --mode eval --episodes 35 36 37 --ood_red --save_video
  ```
- **Train $\pi_{0.5}$ via LeRobot (24 GB L4 / A100):**
  ```bash
  python scripts/pi05_runner.py --mode train --batch_size 8 --steps 5000
  ```
- **Record 4-Camera MuJoCo LeRobot Dataset + Side-by-Side Comparison:**
  ```bash
  python scripts/record_sim_dataset.py --episodes 1 --output_dir ./sim_aloha_ep1 --save_preview
  ```
- **Interactive 3D Desktop Viewer (Local GUI):**
  ```bash
  python scripts/sim/live_viewer.py --episode 1
  ```
- **Generate Action-Preserving Augmentations / V2V Diffusion Clips:**
  ```bash
  python scripts/augmentation/generate_aloha_augmented_dataset.py --episodes 0 1 2
  python scripts/augmentation/export_aloha_v2v_clips.py --mode both
  ```
