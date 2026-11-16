"""
Interactive Real-Time 3D Desktop Viewer for ALOHA Simulation
With:
1. Exact Gripper Actuator Calibration (maps dataset servo radians to MuJoCo linear slide meters)
2. High-friction grip surfaces so screwdriver does not slip or drop
3. Exact Blue Paper Cup with White Rim & Yellow Sponge inside
"""

import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

SCENE_XML = r"E:\mediapipe\physical_ai_eval\mujoco_menagerie\aloha\scene_with_screwdriver.xml"
PARQUET_FILE = r"E:\mediapipe\robotics_datasets\aloha_static_screw_driver\data\chunk-000\file-000.parquet"

print("=" * 65)
print("🤖 ALOHA DIGITAL TWIN: CALIBRATED GRIPPER & OBJECT PHYSICS")
print("=" * 65)

print("📂 Loading Episode 0 telemetry...")
df = pd.read_parquet(PARQUET_FILE)
ep0 = df[df["episode_index"] == 0].sort_values("frame_index")
actions = [np.array(row["action"], dtype=np.float32) for _, row in ep0.iterrows()]
initial_state = np.array(ep0["observation.state"].iloc[0], dtype=np.float32)

print("⚙️ Loading DeepMind ALOHA Model into MuJoCo...")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
screw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
screw_qpos_adr = model.jnt_qposadr[screw_joint_id]

def scale_gripper(action_val):
    """
    Maps real ALOHA Dynamixel servo horn angle (~0.0 to 0.69 rad)
    to MuJoCo linear slide distance (0.002m closed to 0.037m open).
    """
    norm = np.clip(action_val / 0.69, 0.0, 1.0)
    return float(0.002 + norm * (0.037 - 0.002))

def map_action_to_ctrl(action_14):
    ctrl = np.zeros(14, dtype=np.float32)
    # Left Arm (6 joints) + Left Gripper (1 linear slide)
    ctrl[0:6] = action_14[0:6]
    ctrl[6] = scale_gripper(action_14[6])
    # Right Arm (6 joints) + Right Gripper (1 linear slide)
    ctrl[7:13] = action_14[7:13]
    ctrl[13] = scale_gripper(action_14[13])
    return ctrl

def reset_simulation(model, data, state_14):
    mujoco.mj_resetData(model, data)
    
    # 1. Reset ALOHA Arm Joint Angles & Gripper Fingers
    data.qpos[0:6] = state_14[0:6]
    grip_l_m = scale_gripper(state_14[6]) / 2.0
    data.qpos[6] = grip_l_m
    data.qpos[7] = grip_l_m

    data.qpos[8:14] = state_14[7:13]
    grip_r_m = scale_gripper(state_14[13]) / 2.0
    data.qpos[14] = grip_r_m
    data.qpos[15] = grip_r_m

    data.ctrl[:14] = map_action_to_ctrl(state_14)

    # 2. Reset Screwdriver lying flat directly on the table [x, y, z, qw, qx, qy, qz]
    data.qpos[screw_qpos_adr : screw_qpos_adr + 7] = [0.05, -0.10, 0.012, 0.7071, 0.0, 0.7071, 0.0]
    
    mujoco.mj_forward(model, data)

reset_simulation(model, data, initial_state)

print("\n🚀 LAUNCHING 3D INTERACTIVE WINDOW...")
print("=" * 65)
print("🖱️ Mouse Controls:")
print("   • Left Click + Drag  : Orbit / Rotate 3D Camera")
print("   • Right Click + Drag : Zoom in / out")
print("   • Middle Click / Scroll: Pan camera")
print("   • Spacebar: Pause / Resume")
print("=" * 65)

initial_screw_z = data.xpos[screw_body_id][2]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Reset to initial demonstration pose
        reset_simulation(model, data, initial_state)
        viewer.sync()
        time.sleep(0.5)

        for frame_idx, action in enumerate(actions):
            if not viewer.is_running():
                break

            # Send calibrated 14 motor target commands
            data.ctrl[:14] = map_action_to_ctrl(action)

            # Step physics
            for _ in range(4):
                mujoco.mj_step(model, data)

            current_screw_z = data.xpos[screw_body_id][2]
            lift_delta = current_screw_z - initial_screw_z

            viewer.sync()
            time.sleep(1.0 / 50.0)  # 50 Hz real-time playback

        print(f"🔁 Episode finished! Screwdriver peak lift: {lift_delta * 100:.1f} cm. Replaying...")
        time.sleep(1.0)
