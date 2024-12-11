import numpy as np
import mujoco_py

def apply_action(sim, action):
    """
    The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        pos_action, gripper_action = np.split(action, (sim.model.nmocap * 7, ))
        if sim.data.ctrl is not None:
            for i in range(gripper_action.shape[0]):
                sim.data.ctrl[i] = gripper_action[i]
        pos_action = pos_action.reshape(sim.model.nmocap, 7)
        pos_delta, quat_delta = pos_action[:, :3], pos_action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = quat_normalize(quat_delta,sim.data.mocap_quat)

def quat_multiply(quat1, quat2):
    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_normalize(quat,mocap_quat):
    norm = np.linalg.norm(quat)
    if norm == 0:
        return mocap_quat 
    return quat / norm

def robot_get_obs(sim):
    """
    Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        # joint1: qpos[0] 
		# joint2: qpos[1] 
		# joint3: qpos[2] 
		# joint4: qpos[3] 
		# joint5: qpos[4]
		# joint6: qpos[5] 
		# drive_joint: qpos[6] 
		# left_finger_joint: qpos[7] 
		# left_inner_knuckle_joint: qpos[8]
		# right_outer_knuckle_joint: qpos[9] 
		# right_finger_joint: qpos[10]
		# right_inner_knuckle_joint: qpos[11] 
		# object_joint0: qpos[12]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)

def reset(sim):
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0., 0., 0., 1., 0., 0., 0.])
    sim.forward()

def reset_mocap2body_xpos(sim):
    if sim.model.eq_type is None or sim.model.eq_obj1id is None or sim.model.eq_obj2id is None:
        return

    # For all weld constraints
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue
        body2 = sim.model.body_id2name(obj2_id)
        if body2 == 'B0' or body2== 'B9' or body2 == 'B1':
            continue
        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id
        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]