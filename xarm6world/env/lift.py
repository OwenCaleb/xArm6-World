import numpy as np
from xarm6world.env import Base

class Lift(Base):
	def __init__(self,**kwargs):
		self._z_threshold = 0.35
		super().__init__('lift',**kwargs)
		self._init_obj=self.obj

	@property
	def z_target(self):
		return self._init_z + self._z_threshold

	def is_success(self):
		return self.obj[2] >= self.z_target

	def get_reward(self):
		reach_dist = np.linalg.norm(self.obj - self.eef)
		reach_dist_xy = np.linalg.norm(self.obj[:-1] - self.eef[:-1])
		pick_completed = self.obj[2] >= (self.z_target - 0.01)
		obj_dropped = (self.obj[2] < (self._init_z + 0.005)) and (reach_dist > 0.02)

		# Reach
		if reach_dist < 0.05:
			reach_reward = -reach_dist + max(self._action[-1], 0)/50
		elif reach_dist_xy < 0.05:
			reach_reward = -reach_dist
		else:
			z_bonus = np.linalg.norm(np.linalg.norm(self.obj[-1] - self.eef[-1]))
			reach_reward = -reach_dist - 2*z_bonus

		# Pick
		if pick_completed and not obj_dropped:
			pick_reward = self.z_target
		elif (reach_dist < 0.1) and (self.obj[2] > (self._init_z + 0.005)):
			pick_reward = min(self.z_target, self.obj[2])
		else:
			pick_reward = 0

		return reach_reward/10 + pick_reward*10

	def _get_obs(self):
		eef_velp = self.sim.data.get_site_xvelp('grasp') * self.dt
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		link6_mocap_quat=self.link6_mocap_quat


		obj_rot = self.sim.data.get_joint_qpos('object_joint0')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('object_site') * self.dt
		obj_velr = self.sim.data.get_site_xvelr('object_site') * self.dt
		
		obs = np.concatenate([
			self.eef,np.array([0.,0.,0.,0.]) if self.control_mode == 'simple' else np.array([]) , eef_velp, self.obj, obj_rot, obj_velp, obj_velr, self.eef[:3]-self.obj,link6_mocap_quat,
			np.array([
				np.linalg.norm(self.eef[:3]-self.obj), np.linalg.norm(self.eef[:2]-self.obj[:-1]),
				self.z_target, self.z_target-self.obj[-1], self.z_target-self.eef[2],
				gripper_angle
			])
		], axis=0)
		return dict(
			observation=obs,
			state=np.concatenate([self.eef,np.array([gripper_angle,])]),
			achieved_goal=self.eef[0:3],
			desired_goal=self.obj
		)

	def _sample_goal(self):
		# Gripper
		current_mocap_pos = self.sim.data.get_mocap_pos('robot0:mocap2')
		self.sim.data.set_mocap_pos('robot0:mocap2', current_mocap_pos+self.np_random.uniform(-0.05, 0.05, size=3))

		# Object
		object_pos = self.sim.data.get_site_xpos('object_site')
		y_range=0.76*0.5*1.414
		x_range=0.76*0.5*1.414
		object_pos[0]  = self.np_random.uniform(0.15, x_range, size=1)
		object_pos[1]  = self.np_random.uniform(-y_range, y_range, size=1)
		object_qpos = self.sim.data.get_joint_qpos('object_joint0')
		object_qpos[:3] = object_pos
		self.sim.data.set_joint_qpos('object_joint0', object_qpos)
		self._init_z = object_pos[2]
		# Goal
		return object_pos + np.array([0, 0, self._z_threshold])
	
	def reset(self):
		self._action = np.zeros(4)  
		if self.control_mode == 'simple':
			return super().reset()
		elif self.control_mode == 'complex':
			self.quat = np.array([0., 1., 0., 0.])  
			self._action = np.concatenate([self._action[:3], self.quat, [self._action[3]]])  # 拼接
			return super().reset()
		
	def step(self, action):
		self._action = action.copy()
		return super().step(action)
