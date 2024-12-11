"""
Author: OpenAI
Modified by: Wenbo Li
Institution: SCUT
"""
import os
import numpy as np
import glfw
from xarm6world.env import robot_env,mocap
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Base(robot_env.RobotEnv):
	"""
	Superclass for all simxarm environments.
	"""


	def __init__(
			self,
			xml_name='lift',
			action_control_mode='simple', 
			gripper_rotation=[0,1,0,0]
			):
		self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)
		self.center_of_table = np.array([0.5, 0, 0.5])
		self.mode2camera = {
            "mode1": ['camera1'],      # arm
            "mode2": ['camera0'],      # remote
            "mode3": ['camera2'],      # upview
            "mode4": ['camera3'],      # front
            "mode5": ['camera1', 'camera0'],   # arm + remote
            "mode6": ['camera1', 'camera2'],   # arm + upview
            "mode7": ['camera1', 'camera3'],   # arm + front
            "mode8": ['camera0', 'camera2'],   # remote + upview
            "mode9": ['camera0', 'camera3'],   # remote + front
            "mode10": ['camera2', 'camera3'],  # upview + front
            "mode11": ['camera1', 'camera0', 'camera2'],  # arm + remote + upview
            "mode12": ['camera1', 'camera0', 'camera3'],  # arm + remote + front
            "mode13": ['camera1', 'camera2', 'camera3'],  # arm + upview + front
            "mode14": ['camera0', 'camera2', 'camera3'],  # remote + upview + front
            "mode15": ['camera1', 'camera0', 'camera2', 'camera3'],  # all cameras
        }
		self.max_z = 1.1
		self.min_z = 0.54
		self.control_mode=action_control_mode
		super().__init__(
			model_path=os.path.join(os.path.dirname(__file__), 'assets', xml_name + '.xml'),# lift.xml
			n_substeps=20, n_actions=8 if self.control_mode=='complex' else 4, initial_qpos={}
		)

	@property
	def dt(self):
		return self.sim.nsubsteps * self.sim.model.opt.timestep

	@property
	def eef(self):
		'''
		Return the position of the site element 'grasp' in the global coordinate system.
		Depending on the control mode, either 'simple' or 'complex', return the appropriate data.
		'''
		if self.control_mode == 'simple':
			# For simple control, return position and gripper only
			return self.sim.data.get_site_xpos('grasp')
		elif self.control_mode == 'complex':
			# For complex control, return position, orientation (quaternion), and gripper
			pos = self.sim.data.get_site_xpos('grasp')
			grasp_quat = self.sim.data.get_body_xquat('link6')
			return np.concatenate([pos, grasp_quat])
		else:
			raise ValueError("Unknown control action mode: {}".format(self.control_mode))
	
	
	@property
	def link6_mocap_quat(self):
		'''
		Returns the quaternion representing the orientation of the mocap linked to 'link6' in the global coordinate system.
		'''
		body_id = self.sim.model.body_name2id('robot0:mocap2')
		return self.sim.data.body_xquat[body_id]

	@property
	def obj(self):
		'''
		The return value is the position of the site element 'object_site' in the global coordinate system.
		'''
		return self.sim.data.get_site_xpos('object_site')
	@property
	def target(self):
		'''
		The return value is the position of the site element 'target0' in the global coordinate system.
		'''
		return self.sim.data.get_site_xpos('target0')
	@property
	def robot_state(self):
		# Right_outer_knuckle_joint for Single control (1 : closed -1 : open)
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')
		return np.concatenate([self.eef, [gripper_angle]])

	def is_success(self):
		return NotImplementedError()
	
	def get_reward(self):
		raise NotImplementedError()
	
	def _sample_goal(self):
		raise NotImplementedError()

	def get_obs(self):
		return self._get_obs()

	def _step_callback(self):
		self.sim.forward()

	def _limit_gripper(self, gripper_pos, pos_ctrl):
		if gripper_pos[0] > 0.76*0.5*1.414:
			pos_ctrl[0] = min(pos_ctrl[0], 0)
		if gripper_pos[0] < self.center_of_table[0] -0.35:
			pos_ctrl[0] = max(pos_ctrl[0], 0)
		if gripper_pos[1] > self.center_of_table[1] + 0.76*0.5*1.414:
			pos_ctrl[1] = min(pos_ctrl[1], 0)
		if gripper_pos[1] < self.center_of_table[1] - 0.76*0.5*1.414:
			pos_ctrl[1] = max(pos_ctrl[1], 0)
		if gripper_pos[2] > self.max_z:
			pos_ctrl[2] = min(pos_ctrl[2], 0)
		if gripper_pos[2] < self.min_z:
			pos_ctrl[2] = max(pos_ctrl[2], 0)
		return pos_ctrl

	def _quat_normalize(self,quat):
		norm = np.linalg.norm(quat)
		if norm == 0:
			return np.array([0, 1, 0, 0]) 
		return quat / norm
	
	def _slerp(self,quat1, quat2, n):
		quat1 = quat1 / np.linalg.norm(quat1)
		quat2 = quat2 / np.linalg.norm(quat2)
		
		quat1 = np.concatenate([quat1[1:] ,quat1[:1]])
		quat2 = np.concatenate([quat2[1:] ,quat2[:1]])
		times = np.array([0.0, 1.0])
		quat1 = R.from_quat(quat1)
		quat2 = R.from_quat(quat2)
		slerp = Slerp(times, R.concatenate([quat1, quat2]))
		t = 1.0 / n
		quat_result = slerp(t).as_quat()
		result = np.concatenate([quat_result[3:] ,quat_result[:3]])
		return result
	
	def _quat_multiply(self,quat1, quat2):
		w1, x1, y1, z1 = quat1
		w2, x2, y2, z2 = quat2
		
		w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
		x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
		y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
		z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
		return np.array([w, x, y, z])
	
	def _apply_action(self, action):
		if self.control_mode=='simple':
			assert action.shape == (4,)
			action = action.copy()
			pos_ctrl, gripper_ctrl = action[:3], action[3]
		else:
			assert action.shape == (8,)
			action = action.copy()
			pos_ctrl, self.gripper_rotation, gripper_ctrl = action[:3], action[3:7], action[7]
			self.gripper_rotation = self._quat_normalize(self.gripper_rotation)
			current_quat = self.sim.data.mocap_quat[:] 
			target_quat = self._quat_multiply(current_quat[0], self.gripper_rotation)  
			self.gripper_rotation = self._slerp(current_quat[0], target_quat, self.sim.nsubsteps)
			
		pos_ctrl = self._limit_gripper(self.sim.data.get_site_xpos('grasp'), pos_ctrl) * (1/self.sim.nsubsteps)
		gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
		mocap.apply_action(self.sim, np.concatenate([pos_ctrl, self.gripper_rotation, gripper_ctrl]))


	def _viewer_setup(self):
		body_id = self.sim.model.body_name2id('link6')
		lookat = self.sim.data.body_xpos[body_id]
		camera_id = self.sim.model.camera_name2id("camera0")  
		self.viewer.cam.trackbodyid = camera_id 
		for idx, value in enumerate(lookat):
			self.viewer.cam.lookat[idx] = value
		self.viewer.cam.distance = 4.0
		self.viewer.cam.azimuth = 132.
		self.viewer.cam.elevation = -14.

	def _render_callback(self):
		self.sim.forward()

	def _reset_sim(self):
		
		self.sim.set_state(self.initial_state)
		# mocap_name = 'robot0:mocap2'  # mocap body 名称
		# mocap_index = self.sim.model.body_name2id(mocap_name)  # 获取 mocap body 的索引
		# mocap_xpos = self.sim.data.body_xpos[mocap_index]  # 获取 mocap body 的位置 (x, y, z)
		# print(mocap_xpos)
		# body_index = self.sim.model.body_name2id('link6')  # 获取 body 的索引
		# body_xpos = self.sim.data.body_xpos[body_index]  # 获取 body 的位置 (x, y, z)
		# print(body_xpos)
		# print(self.sim.data.get_site_xpos('grasp'))
		self._sample_goal()
		for _ in range(100):
			self.sim.step()

		# mocap_name = 'robot0:mocap2'  # mocap body 名称
		# mocap_index = self.sim.model.body_name2id(mocap_name)  # 获取 mocap body 的索引
		# mocap_xpos = self.sim.data.body_xpos[mocap_index]  # 获取 mocap body 的位置 (x, y, z)
		# body_index = self.sim.model.body_name2id('link6')  # 获取 body 的索引
		# body_xpos = self.sim.data.body_xpos[body_index]  # 获取 body 的位置 (x, y, z)
		# print(mocap_xpos)
		# print(body_xpos)
		# print(self.sim.data.get_site_xpos('grasp'))
		return True

	def _set_gripper(self, gripper_pos, gripper_rotation):
		self.sim.data.set_mocap_pos('robot0:mocap2', gripper_pos)
		self.sim.data.set_mocap_quat('robot0:mocap2', gripper_rotation)
		self.sim.data.set_joint_qpos('right_outer_knuckle_joint', 0)
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
		self.sim.data.qpos[8] = 0.0
		self.sim.data.qpos[11] = 0.0

	def _env_setup(self, initial_qpos):
		for name, value in initial_qpos.items():
			self.sim.data.set_joint_qpos(name, value)
		mocap.reset(self.sim)
		self.sim.forward()
		self._sample_goal()
		self.sim.forward()

	def reset(self):
		self._reset_sim()
		return self._get_obs()

	def step(self, action):
		if self.control_mode=='simple':
			assert action.shape == (4,)
		else:
			assert action.shape == (8,)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		self._apply_action(action)
		for _ in range(4):
			self.sim.step()
		self._step_callback()
		obs = self._get_obs()
		reward = self.get_reward()
		done = False
		info = {'is_success': self.is_success(), 'success': self.is_success()}
		return obs, reward, done, info
	
	def render(self,mode='rgb_array',camera_name='camera0',width=224,height=224):
		self._render_callback()
		viewer = self._get_viewer(mode)
		if mode == 'rgb_array':
			rgb_image = self.sim.render(width, height, camera_name=camera_name, depth=False)[::-1, :, :]
			return rgb_image
		elif mode == 'human':
			viewer.cam.fixedcamid = self.sim.model.camera_name2id(camera_name)
			self.viewer.render()
			return None

	def obs(self, mode='rgb', width=84, camera_mode='mode1'):
		self._render_callback()
		result = {}
		if mode in ['rgb', 'all']:
			rgb_images = {}
			camera_modes = self.mode2camera.get(camera_mode, [])
			for camera in camera_modes:
				rgb_image = self.sim.render(width, width, camera_name=camera, depth=False)[::-1, :, :]
				rgb_images[camera] = rgb_image
			result['rgb'] = rgb_images
		if mode in ['depth', 'all']:
			depth_images = {}
			camera_modes = self.mode2camera.get(camera_mode, [])
			for camera in camera_modes:
				depth_image = self.sim.render(width, width, camera_name=camera, depth=True)[1][::-1, :]
				depth_images[camera] = depth_image
			result['depth'] = depth_images
		return result
	
	def close(self):
		if self.viewer is not None:
			# 根据渲染模式判断是否需要销毁窗口
			if hasattr(self.viewer, 'window'):
				# 只有在 human 渲染模式下才有 window 属性
				print("Closing window glfw")
				glfw.destroy_window(self.viewer.window)
			# 清理viewer
			self.viewer = None
		self._viewers = {}
		super().close()
