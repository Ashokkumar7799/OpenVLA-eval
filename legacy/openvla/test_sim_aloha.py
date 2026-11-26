"""
ALOHA MuJoCo Simulation Rollout & Trajectory Executor
Renders an MP4 video of the simulation rollout with the 3D Screwdriver & Cup.
"""

import os
import time
import numpy as np
import pandas as pd
import cv2
import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "aloha", "scene_with_screwdriver.xml")
PARQUET_FILE = os.environ.get(
    "ALOHA_PARQUET_FILE",
    os.path.join(os.path.dirname(REPO_ROOT), "robotics_datasets", "aloha_static_screw_driver", "data", "chunk-000", "file-000.parquet"),
)
OUTPUT_VIDEO = os.path.join(REPO_ROOT, "scratch", "sim_rollout_with_screwdriver.mp4")

print("=" * 65)
print("🤖 ALOHA SIMULATION VIDEO RECORDER (WITH SCREWDRIVER & CUP)")
print("=" * 65)

# 1. Load Episode 0 telemetry
df = pd.read_parquet(PARQUET_FILE)
ep0 = df[df["episode_index"] == 0].sort_values("frame_index")
num_frames = len(ep0)
initial_state = np.array(ep0["observation.state"].iloc[0], dtype=np.float32)

# 2. Compile Model
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

screw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "screwdriver_joint")
screw_qpos_adr = model.jnt_qposadr[screw_joint_id]

# Reset initial positions
mujoco.mj_resetData(model, data)
data.qpos[0:6] = initial_state[0:6]
data.qpos[6] = initial_state[6] / 2.0
data.qpos[7] = initial_state[6] / 2.0
data.qpos[8:14] = initial_state[7:13]
data.qpos[14] = initial_state[13] / 2.0
data.qpos[15] = initial_state[13] / 2.0
data.ctrl[:14] = initial_state[:14]
data.qpos[screw_qpos_adr : screw_qpos_adr + 7] = [-0.08, -0.02, 0.265, 1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

# 3. Setup Renderer (Overhead camera view)
renderer = mujoco.Renderer(model, height=480, width=640)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 50.0, (640, 480))

print("🚀 Recording 3D simulation video...")
for i, (_, row) in enumerate(ep0.iterrows()):
    action_14 = np.array(row["action"], dtype=np.float32)
    data.ctrl[:14] = action_14[:14]

    for _ in range(4):
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="overhead_cam")
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr)

video_writer.release()
print(f"✅ Video recorded and saved to: {OUTPUT_VIDEO}")
