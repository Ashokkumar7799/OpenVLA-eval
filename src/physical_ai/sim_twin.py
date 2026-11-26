"""Calibrated 14-DoF ALOHA MuJoCo Digital Twin kinematics, contact physics, and visual alignment."""

import os
import sys
import numpy as np

ACTION_MIN = np.array([
    -0.0276, -0.9986,  0.9311, -0.4126, -1.1950, -0.2286, -0.0092,
    -0.2807, -0.9710, -0.0107, -0.1488, -0.9403, -0.3007, -0.0834,
], dtype=np.float32)

ACTION_MAX = np.array([
     0.5354,  0.0046,  1.3530,  0.1227, -0.3037,  0.4034,  0.7713,
     0.2255,  0.5216,  1.1796,  0.1089,  0.7194,  0.3252,  0.7697,
], dtype=np.float32)

# Empirical ALOHA gripper calibration from dataset stats:
# Action gripper commands span ~[0.0, 0.75] rad; State gripper feedback spans ~[0.0, 0.84] rad.
ACTION_GRIPPER_OPEN_RAD = 0.75
STATE_GRIPPER_OPEN_RAD = 0.84

CAM_MAP = {
    "cam_high": "overhead_cam",
    "cam_low": "worms_eye_cam",
    "cam_left_wrist": "wrist_cam_left",
    "cam_right_wrist": "wrist_cam_right",
}

DEFAULT_TASK = "Pick up the screwdriver with the right arm, hand it over to the left arm then place it into the cup."


def ensure_mujoco_gl():
    """Enables headless EGL rendering on Linux/Colab servers when no display is attached."""
    if sys.platform != "win32" and "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


def scale_gripper(a_grip: float) -> float:
    """Converts ALOHA gripper command radians into MuJoCo linear slide joint meters."""
    norm = np.clip(a_grip / ACTION_GRIPPER_OPEN_RAD, 0.0, 1.0)
    return float(0.002 + norm * (0.037 - 0.002))


def mujoco_to_gripper_rad(pos_m: float) -> float:
    """Converts MuJoCo single-finger slide meters back to ALOHA state radian scale."""
    norm = np.clip((pos_m - 0.002) / (0.037 - 0.002), 0.0, 1.0)
    return float(norm * STATE_GRIPPER_OPEN_RAD)


def map_action_to_ctrl(act) -> np.ndarray:
    """Maps a 14-DoF dataset/policy action vector into MuJoCo actuator control targets."""
    ctrl = np.zeros(14, dtype=np.float32)
    ctrl[0:6] = act[0:6]
    ctrl[6] = scale_gripper(act[6])
    ctrl[7:13] = act[7:13]
    ctrl[13] = scale_gripper(act[13])
    return ctrl


def read_aloha_state_14(data) -> np.ndarray:
    """Extracts the 14-DoF proprioceptive state vector matching LeRobot observation.state."""
    state = np.zeros(14, dtype=np.float32)
    state[0:6] = data.qpos[0:6]
    state[6] = mujoco_to_gripper_rad(data.qpos[6])
    state[7:13] = data.qpos[8:14]
    state[13] = mujoco_to_gripper_rad(data.qpos[14])
    return state


def apply_visual_alignment(model, shadow_size: int = 1024):
    """Aligns MuJoCo camera FOVs, studio backdrop, and white grippers with real ALOHA frames."""
    import mujoco

    for g_id in range(model.ngeom):
        m_id = model.geom_dataid[g_id] if model.geom_type[g_id] == mujoco.mjtGeom.mjGEOM_MESH else -1
        if m_id >= 0:
            m_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m_id) or ""
            if any(k in m_name for k in ["custom_finger", "gripper_wrist_mount"]):
                model.geom_matid[g_id] = -1
                model.geom_rgba[g_id] = [0.84, 0.85, 0.87, 1.0]

    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_gid != -1:
        model.geom_matid[floor_gid] = -1
        model.geom_rgba[floor_gid] = [0.68, 0.69, 0.71, 1.0]
        model.geom_size[floor_gid] = [8.0, 8.0, 0.05]

    model.vis.rgba.haze[:] = [0.80, 0.81, 0.82, 1.0]
    if model.ntex > 0:
        for t_id in range(model.ntex):
            if model.tex_type[t_id] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                adr = model.tex_adr[t_id]
                w, h = model.tex_width[t_id], model.tex_height[t_id]
                nchan = model.tex_nchannel[t_id]
                sz = w * h * nchan
                model.tex_data[adr:adr + sz] = 215

    cid_high = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
    if cid_high != -1:
        model.cam_sensorsize[cid_high] = [0.0, 0.0]
        model.cam_fovy[cid_high] = 35.5
        ang = np.radians(15.5)
        model.cam_quat[cid_high] = [np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0]

    for wc in ("wrist_cam_left", "wrist_cam_right"):
        cid_w = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, wc)
        if cid_w != -1:
            model.cam_sensorsize[cid_w] = [0.0, 0.0]
            model.cam_fovy[cid_w] = 48.0

    cid_low = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "worms_eye_cam")
    if cid_low != -1:
        model.cam_sensorsize[cid_low] = [0.0, 0.0]
        model.cam_fovy[cid_low] = 48.0
        model.cam_pos[cid_low] = [-0.03, -0.34, 0.56]
        target = np.array([-0.03, 0.25, 0.0])
        fwd = target - model.cam_pos[cid_low]
        fwd = fwd / np.linalg.norm(fwd)
        up_hint = np.array([1.0, 0.0, 0.0])
        z_cam = -fwd
        x_cam = np.cross(up_hint, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= np.linalg.norm(y_cam)
        rot = np.column_stack([x_cam, y_cam, z_cam])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, rot.flatten())
        model.cam_quat[cid_low] = q

    model.vis.quality.shadowsize = int(shadow_size)


def apply_ood_red_theme(model):
    """Recolors screwdriver and cup materials to red for out-of-distribution robustness tests."""
    import mujoco

    for mat_name, rgba in [
        ("screwdriver_handle", [0.88, 0.14, 0.14, 1.0]),
        ("cup_blue_outer", [0.82, 0.12, 0.12, 1.0]),
        ("cup_sponge_yellow", [0.78, 0.20, 0.20, 1.0]),
    ]:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
        if mid != -1:
            model.mat_rgba[mid] = rgba


def calibrate_screwdriver_contact(model, torsional_friction: float = 0.010):
    """Lowers torsional friction on the screwdriver so gravity gently levels it for handover."""
    import mujoco

    for gname in ("screwdriver_grip", "screwdriver_shaft", "screwdriver_tip"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid != -1:
            model.geom_friction[gid, 1] = float(torsional_friction)


def compute_episode_grasp_point(model, data, actions, x_offset: float = -0.012):
    """Finds the tabletop (x, y) grasp origin from right-gripper fingertip sites via FK."""
    import mujoco

    s_finger_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/left_finger")
    s_finger_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/right_finger")
    use_site = s_finger_l != -1 and s_finger_r != -1
    if not use_site:
        b_finger_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/left_finger_link")
        b_finger_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/right_finger_link")

    z_list = []
    pos_list = []
    for f in range(min(220, len(actions))):
        act = actions[f]
        data.qpos[0:6] = act[0:6]
        data.qpos[6] = data.qpos[7] = scale_gripper(act[6])
        data.qpos[8:14] = act[7:13]
        data.qpos[14] = data.qpos[15] = scale_gripper(act[13])
        mujoco.mj_forward(model, data)
        if use_site:
            pt = (data.site_xpos[s_finger_l] + data.site_xpos[s_finger_r]) / 2.0
        else:
            pt = (data.xpos[b_finger_l] + data.xpos[b_finger_r]) / 2.0
        z_list.append(pt[2])
        pos_list.append(pt.copy())

    min_f = int(np.argmin(z_list))
    gx = float(pos_list[min_f][0]) + float(x_offset)
    gy = float(pos_list[min_f][1])
    return gx, gy


def reset_aloha_episode(model, data, initial_state, gx: float, gy: float, screw_joint_id: int):
    """Resets ALOHA joints, grippers, controls, and spawns the screwdriver at (gx, gy)."""
    import mujoco

    mujoco.mj_resetData(model, data)
    data.qpos[0:6] = initial_state[0:6]
    data.qpos[6] = data.qpos[7] = scale_gripper(initial_state[6])
    data.qpos[8:14] = initial_state[7:13]
    data.qpos[14] = data.qpos[15] = scale_gripper(initial_state[13])
    data.ctrl[:14] = map_action_to_ctrl(initial_state)

    if screw_joint_id != -1:
        qadr = model.jnt_qposadr[screw_joint_id]
        data.qpos[qadr:qadr + 7] = [gx, gy, 0.015, 0.7071, 0.0, 0.7071, 0.0]

    mujoco.mj_forward(model, data)


def evaluate_episode_outcome(model, data, initial_z: float, max_z: float, screw_body_id: int, cup_body_id: int):
    """Computes physical lift height, cup proximity, and binary success flags."""
    lift_height_cm = float((max_z - initial_z) * 100.0)
    lift_success = bool(lift_height_cm >= 5.0)
    final_pos = data.xpos[screw_body_id] if screw_body_id != -1 else np.zeros(3)
    cup_pos = model.body_pos[cup_body_id] if cup_body_id != -1 else np.array([-0.12, 0.19, 0.001])
    cup_dist_cm = float(np.linalg.norm(final_pos[:2] - cup_pos[:2]) * 100.0)
    cup_success = bool(cup_dist_cm < 10.0 and final_pos[2] < 0.19)
    return lift_height_cm, lift_success, cup_dist_cm, cup_success
