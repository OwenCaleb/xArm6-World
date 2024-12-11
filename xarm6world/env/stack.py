import numpy as np
from xarm6world.env import Base

class Stack(Base):
	def __init__(self,**kwargs):
		self._z_threshold = 0.07  # 堆叠高度阈值，可以根据需要调整
		self.reward_scale = 100
		super().__init__('stack', **kwargs)
		self._init_obj = self.obj  # 初始物体位置
		self._init_obj1 = self.sim.data.get_site_xpos('object_site1')

	def is_success(self):
		obj_pos = self.obj
		obj1_pos = self.sim.data.get_site_xpos('object_site1')
		return (obj_pos[2] >= obj1_pos[2] + self._z_threshold) and self._is_above(obj_pos, obj1_pos)
	
	def _is_above(self, obj, obj1, xy_threshold=0.05):
		return np.linalg.norm(obj[:2] - obj1[:2]) < xy_threshold
	
	
	
	def get_reward(self, action=None):
		"""
        Reward function for the Stack task.

        Sparse un-normalized reward:
            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:
            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:
            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        The final reward is normalized and scaled by reward_scale / 2.0 so that the max score is equal to reward_scale
        """
		obj_pos = self.obj
		obj1_pos = self.sim.data.get_site_xpos('object_site1')
	
		if self.control_mode == 'simple':
			reach_dist = np.linalg.norm(obj_pos - self.eef)
		elif self.control_mode == 'complex':
			reach_dist = np.linalg.norm(obj_pos[:3] - self.eef[:3])
		reaching = max(0.25 - reach_dist, 0) / 0.25  # Normalize to [0,1] and scale to [0, 0.25]
		
		grasping = 0.25 if self._is_grasping() else 0.0
		
		
		lifting = 1.0 if obj_pos[2] > self._init_obj[2] + 0.07 else 0.0 
		
		aligning = max(0.5 - np.linalg.norm(obj_pos[:2] - obj1_pos[:2]), 0) / 0.5  # Normalize to [0,1] and scale to [0, 0.5]
		
		stacking = 2.0 if self.is_success() else 0.0
		
		
		shaped_reward1 = reaching + grasping  # Max 0.25 + 0.25 = 0.5
		shaped_reward2 = lifting + aligning    # Max 1.0 + 0.5 = 1.5
		shaped_reward3 = stacking             # Max 2.0
		
		shaped_reward = max(shaped_reward1, shaped_reward2, shaped_reward3)
		
		sparse_reward = stacking
		
		total_reward = max(shaped_reward, sparse_reward)

        # Normalize and scale
		normalized_reward = (total_reward ) * self.reward_scale
		return normalized_reward
	
	
	
	def _is_grasping(self):
		grasp_threshold = 0.05  # 抓取阈值距离
		obj_pos = self.obj
		gripper_pos = self.eef[:3]  # 假设 eef 是夹爪的位置
		distance = np.linalg.norm(obj_pos - gripper_pos)
		# env.sim.data.get_joint_qpos('left_inner_knuckle_joint')
		# gripper_closed = self.sim.data.get_joint_qpos('left_inner_knuckle_joint') < 0.1  # close 0.8538885804161165 open 0.06757915261149063
		return distance < grasp_threshold
	
	
	
	
	def _get_obs(self):
		eef_velp = self.sim.data.get_site_xvelp('grasp') * self.dt
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')
		link6_mocap_quat = self.link6_mocap_quat
		obj_rot = self.sim.data.get_joint_qpos('object_joint0')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('object_site') * self.dt
		obj_velr = self.sim.data.get_site_xvelr('object_site') * self.dt
		obj1_pos = self.sim.data.get_site_xpos('object_site1')
		obj1_rot = self.sim.data.get_joint_qpos('object_joint1')[-4:]
		obj1_velp = self.sim.data.get_site_xvelp('object_site1') * self.dt
		obj1_velr = self.sim.data.get_site_xvelr('object_site1') * self.dt
		obs = np.concatenate([
            self.eef,
            np.array([0., 0., 0., 0.]) if self.control_mode == 'simple' else np.array([]),
            eef_velp,
            self.obj,
            obj_rot,
            obj_velp,
            obj_velr,
            obj1_pos,
            obj1_rot,
            obj1_velp,
            obj1_velr,
            self.eef[:3] - self.obj[:3],
            self.eef[:3] - obj1_pos[:3],
            link6_mocap_quat,
            np.array([
                np.linalg.norm(self.eef[:3] - self.obj[:3]),
                np.linalg.norm(self.eef[:2] - self.obj[:2]),
                np.linalg.norm(self.eef[:3] - obj1_pos[:3]),
                np.linalg.norm(self.eef[:2] - obj1_pos[:2]),
                self._z_threshold,
                self._z_threshold - (self.obj[2] - self._init_obj1[2]),
                self._z_threshold - (self.eef[2] - self._init_obj1[2]),
                gripper_angle
            ])
        ], axis=0)
		return dict(
            observation=obs,
            state=np.concatenate([self.eef, np.array([gripper_angle, ])]),
            achieved_goal=self.obj[:3],
            desired_goal=self._init_obj1[:3] + np.array([self._z_threshold, 0, 0])  # 根据需求调整
        )
	
	
	def _sample_goal(self):
		# Gripper
		current_mocap_pos = self.sim.data.get_mocap_pos('robot0:mocap2')
		self.sim.data.set_mocap_pos('robot0:mocap2', current_mocap_pos+self.np_random.uniform(-0.05, 0.05, size=3))

		object_pos = self.sim.data.get_site_xpos('object_site')
		target_pos = object_pos.copy()
		y_range=0.76*0.5*1.414
		x_range=0.76*0.5*1.414
		object_pos[0]  = self.np_random.uniform(0.15, x_range, size=1)
		object_pos[1]  = self.np_random.uniform(-y_range, y_range, size=1)
		target_pos[0]  = self.np_random.uniform(0.15, x_range, size=1)
		target_pos[1]  = self.np_random.uniform(-y_range, y_range, size=1)
		object_qpos = self.sim.data.get_joint_qpos('object_joint0')
		object_qpos[:3] = object_pos
		target_qpos = self.sim.data.get_joint_qpos('object_joint1')
		target_qpos[:3] = target_pos
		self.sim.data.set_joint_qpos('object_joint0', object_qpos)
		self.sim.data.set_joint_qpos('object_joint1', target_qpos)
		self._init_obj = self.obj 
		self._init_obj1 = target_pos.copy()
		# Goal
		return target_pos + np.array([0, 0, self._z_threshold])
	
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
