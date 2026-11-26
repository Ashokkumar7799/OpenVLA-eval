#!/usr/bin/env python3
"""
Interactive Real-Time 3D Desktop Viewer for ALOHA Digital Twin
=============================================================
Uses calibrated grasp-site initialization, torsional screwdriver friction,
single-finger slide mapping, and exact 50 Hz (10-substep) physics replay.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from physical_ai.data import load_aloha_dataset
from physical_ai.paths import SCENE_XML
from physical_ai.sim_twin import (
    apply_visual_alignment,
    calibrate_screwdriver_contact,
    compute_episode_grasp_point,
    evaluate_episode_outcome,
    map_action_to_ctrl,
    reset_aloha_episode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ALOHA 3D Interactive Simulation Viewer")
    parser.add_argument("--episode", type=int, default=1, help="Episode index to replay (default: 1)")
    parser.add_argument("--no_visual_align", action="store_true", help="Disable Sim-to-Real visual alignment")
    return parser.parse_args()


def main():
    args = parse_args()
    import mujoco
    import mujoco.viewer

    print("=" * 70)
    print(f"🤖 ALOHA DIGITAL TWIN VIEWER: EPISODE {args.episode}")
    print("=" * 70)

    df, _ = load_aloha_dataset(need_videos=False)
    ep_df = df[df["episode_index"] == args.episode].sort_values("frame_index")
    actions = np.array([row["action"] for _, row in ep_df.iterrows()], dtype=np.float32)
    initial_state = np.array(ep_df["observation.state"].iloc[0], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    calibrate_screwdriver_contact(model)
    if not args.no_visual_align:
        apply_visual_alignment(model)

    screw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
    screw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")

    gx, gy = compute_episode_grasp_point(model, data, actions)
    reset_aloha_episode(model, data, initial_state, gx, gy, screw_jid)
    init_z = float(data.xpos[screw_bid][2])

    print(f"🎯 Episode {args.episode} Grasp Origin: X={gx:.4f}m, Y={gy:.4f}m")
    print("🖱️ Left Drag=Orbit | Right Drag=Zoom | Middle=Pan | Space=Pause")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            reset_aloha_episode(model, data, initial_state, gx, gy, screw_jid)
            viewer.sync()
            time.sleep(0.5)

            max_z = init_z
            for act in actions:
                if not viewer.is_running():
                    break
                data.ctrl[:14] = map_action_to_ctrl(act)
                for _ in range(10):
                    mujoco.mj_step(model, data)
                max_z = max(max_z, float(data.xpos[screw_bid][2]))
                viewer.sync()
                time.sleep(1.0 / 50.0)

            lift_cm, lift_ok, cup_dist, cup_ok = evaluate_episode_outcome(
                model, data, init_z, max_z, screw_bid, cup_bid
            )
            print(f"✨ Ep {args.episode}: Lift={lift_cm:.1f}cm ({lift_ok}) | Cup={cup_dist:.1f}cm ({cup_ok})")
            time.sleep(1.5)


if __name__ == "__main__":
    main()
