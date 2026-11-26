"""Cross-platform path resolution for local Windows, Colab, and container deployments."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_DIR = REPO_ROOT / "docs"
DEPLOY_DIR = REPO_ROOT / "deploy"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MUJOCO_DIR = REPO_ROOT / "mujoco_menagerie" / "aloha"

SCENE_XML = str(MUJOCO_DIR / "scene_with_screwdriver.xml")
ALOHA_ASSETS_DIR = str(MUJOCO_DIR / "assets")
SPLITS_JSON = str(CONFIGS_DIR / "dataset_splits.json")
BASELINE_REPORT_JSON = str(CONFIGS_DIR / "baseline_benchmark_report.json")
DEFAULT_HF_REPO = "lerobot/aloha_static_screw_driver"


def resolve_local_dataset_dir() -> str:
    """Returns the local LeRobot dataset directory (env override or sibling robotics_datasets)."""
    env_dir = os.environ.get("ALOHA_DATASET_DIR")
    if env_dir:
        return env_dir
    return str(REPO_ROOT.parent / "robotics_datasets" / "aloha_static_screw_driver")


def resolve_default_augmented_dir() -> str:
    """Returns the default output directory for augmented datasets."""
    env_dir = os.environ.get("ALOHA_AUGMENTED_DIR")
    if env_dir:
        return env_dir
    return str(REPO_ROOT / "augmented_dataset")
