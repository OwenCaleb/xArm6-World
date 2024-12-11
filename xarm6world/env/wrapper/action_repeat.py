import gym
import numpy as np

class ActionRepeatWrapper(gym.Wrapper):
	def __init__(self, env, action, repeat):
		super().__init__(env)
		self._env = env
		self._repeat=repeat
		self.action=action
		self.action_space_dim = self._env.action_space.shape[0]
		self.action_dim_max=4 if self.action_space_dim <= 4 else 8
		self.action_padding = np.zeros(self.action_dim_max - self.action_space_dim, dtype=np.float32)
		if 'w' not in self.action:
			self.action_padding[-1] = 1.0

	def step(self, action):
		action = np.concatenate([action, self.action_padding]) if 'w' not in self.action else action
		reward = 0.0
		for _ in range(self._repeat):
			obs, r, done, info = self._env.step(action)
			reward += r
		return obs, reward, done, info
	
	@property
	def state(self):
		return self._env.robot_state