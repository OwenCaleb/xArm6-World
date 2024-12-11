import numpy as np
from xarm6world.env import Base

class Reach(Base):
	def __init__(self,**kwargs):
		super().__init__('reach',**kwargs)

	def _reset_sim(self):
		self._act_magnitude = 0
		super()._reset_sim()

	def is_success(self):
		return np.linalg.norm(self.eef[:3] - self.goal) <= 0.05

	def get_reward(self):
		dist = np.linalg.norm(self.eef[:3]  - self.goal)
		penalty = self._act_magnitude**2
		return -(dist + 0.15*penalty)*10

	def _get_obs(self):
		eef_velp = self.sim.data.get_site_xvelp('grasp') * self.dt
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		link6_mocap_quat=self.link6_mocap_quat

		eef = np.concatenate([
			self.eef, 
			np.array([0., 0., 0., 0.]) if self.control_mode == 'simple' else np.array([])
		], axis=0)
		obs = np.concatenate([
			eef, eef_velp, self.target, eef[:3]-self.target,link6_mocap_quat,
			np.array([np.linalg.norm(eef[:3]-self.target), gripper_angle])
		], axis=0)
		return dict(
			observation=obs,
			state=np.concatenate([self.eef,np.array([gripper_angle,])]),
			achieved_goal=self.eef[0:3],
			desired_goal=self.target
		)

	def _sample_goal(self):
		# Gripper
		current_mocap_pos = self.sim.data.get_mocap_pos('robot0:mocap2')
		self.sim.data.set_mocap_pos('robot0:mocap2', current_mocap_pos+self.np_random.uniform(-0.05, 0.05, size=3))
		y_range=0.76*0.5*1.414
		x_range=0.76*0.5*1.414
		# Goal
		target=self.target
		target[0]  = self.np_random.uniform(0.15, x_range, size=1)
		target[1]  = self.np_random.uniform(-y_range, y_range, size=1)
		self.sim.model.site_pos[self.sim.model.site_name2id('target0')] =target
		return target

	def step(self, action):
		self._act_magnitude = np.linalg.norm(action[:3])
		return super().step(action)
