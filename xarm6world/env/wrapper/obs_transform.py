import gym
from collections import OrderedDict, deque
import gym
import numpy as np

class ObsTransformWrapper(gym.Wrapper):
	"""
    This wrapper is used to convert the action and observation spaces to the correct format.
	"""
	def __init__(self,env,**kwargs):
		super().__init__(env)
		self._env = env
		self.obs_mode = kwargs.get('obs_mode', 'rgb') 
		self.obs_size = kwargs.get('obs_size', 84) 
		self.render_size = kwargs.get('render_size', 224) 
		self.frame_stack = kwargs.get('frame_stack', 1) 
		self.channel_last = kwargs.get('channel_last', False) 
		self.xarm_camera_mode= kwargs.get('xarm_camera_mode', 'mode2') 
		self.render_mode=kwargs.get('render_mode', 'rgb_array') 
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
		
		assert self.obs_mode in ['state','rgb', 'depth','all'], 'This task only supports state, rgb, depth and all observations.'
		if self.obs_mode in ['rgb', 'depth','all']:
			assert self.xarm_camera_mode in self.mode2camera, 'Invalid camera mode selected. Please select a valid mode (mode1 to mode15).'
		if self.obs_mode  in ['rgb','all']:
			camera_modes = self.mode2camera.get(self.xarm_camera_mode, [])
			if 'camera0' in camera_modes:
				self._frames_cam0_rgb = deque([], maxlen=self.frame_stack)
			if 'camera1' in camera_modes:
				self._frames_cam1_rgb = deque([], maxlen=self.frame_stack)
			if 'camera2' in camera_modes:
				self._frames_cam2_rgb = deque([], maxlen=self.frame_stack)
			if 'camera3' in camera_modes:
				self._frames_cam3_rgb = deque([], maxlen=self.frame_stack)
			image_shape = (self.obs_size, self.obs_size, 3 * self.frame_stack) if self.channel_last else (3 * self.frame_stack, self.obs_size, self.obs_size)
		if self.obs_mode  in ['depth','all']:
			camera_modes = self.mode2camera.get(self.xarm_camera_mode, [])
			if 'camera0' in camera_modes:
				self._frames_cam0_depth = deque([], maxlen=self.frame_stack)
			if 'camera1' in camera_modes:
				self._frames_cam1_depth = deque([], maxlen=self.frame_stack)
			if 'camera2' in camera_modes:
				self._frames_cam2_depth = deque([], maxlen=self.frame_stack)
			if 'camera3' in camera_modes:
				self._frames_cam3_depth = deque([], maxlen=self.frame_stack)
			image_depth_shape = (self.obs_size, self.obs_size, self.frame_stack) if self.channel_last else (self.frame_stack, self.obs_size, self.obs_size)
		if self.obs_mode == 'state':
			self.observation_space = env.observation_space['observation']
		elif self.obs_mode in ['rgb', 'depth', 'all']:
			img_data = {}
			if self.obs_mode in ['rgb', 'all']:
				rgb_space = {}
				camera_modes = self.mode2camera.get(self.xarm_camera_mode, [])
				for camera in camera_modes:
					rgb_space[camera] = gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
				img_data['rgb'] = gym.spaces.Dict(rgb_space)
			if self.obs_mode in ['depth', 'all']:
				depth_space = {}
				camera_modes = self.mode2camera.get(self.xarm_camera_mode, [])
				for camera in camera_modes:
					depth_space[camera] = gym.spaces.Box(low=0., high=1., shape=image_depth_shape, dtype=np.float32)
				img_data['depth'] = gym.spaces.Dict(depth_space)
			self.observation_space = gym.spaces.Dict({
                'img': gym.spaces.Dict(img_data),
                'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            })
		else:
			raise ValueError(f'Unknown obs_mode {self.obs_mode}.')
		
	def _render_obs(self):
		obs = self._env.obs(mode=self.obs_mode, width=self.obs_size,camera_mode=self.xarm_camera_mode)
		processed_obs = {}#
		if 'rgb' in obs:
			processed_obs['rgb']={}
			for cam, rgb_image in obs['rgb'].items():
				# 如果是 channel_last=False（即 CHW 格式），则需要将 HWC 转换为 CHW
				if not self.channel_last:
					processed_obs['rgb'][cam] = rgb_image.transpose(2, 0, 1)
				else:
					processed_obs['rgb'][cam] = rgb_image.copy()
		if 'depth' in obs:
			processed_obs['depth']={}
			for cam, depth_image in obs['depth'].items():
				if not self.channel_last:
					processed_obs['depth'][cam] = np.expand_dims(depth_image, axis=0)  # 添加一个通道维度，(H, W) -> (1, H, W)
				else:
					processed_obs['depth'][cam] = np.expand_dims(depth_image, axis=-1)  # 添加一个通道维度，(H, W) -> (H, W, 1)
		return processed_obs
	def render(self,camera_name='camera0'):
		return self._env.render(mode=self.render_mode,camera_name=camera_name,width=self.render_size,height=self.render_size)

	def _update_frames(self, reset=False):
		image_data = self._render_obs()
		rgb_images=[]
		depth_images=[]
		if 'rgb' in image_data:
			rgb_images = image_data['rgb']
			if 'camera1' in rgb_images:
				self._frames_cam1_rgb.append(rgb_images['camera1'])
			if 'camera0' in rgb_images:
				self._frames_cam0_rgb.append(rgb_images['camera0'])
			if 'camera2' in rgb_images:
				self._frames_cam2_rgb.append(rgb_images['camera2'])
			if 'camera3' in rgb_images:
				self._frames_cam3_rgb.append(rgb_images['camera3'])
		if 'depth' in image_data:
			depth_images = image_data['depth']
			if 'camera1' in depth_images:
				self._frames_cam1_depth.append(depth_images['camera1'])
			if 'camera0' in depth_images:
				self._frames_cam0_depth.append(depth_images['camera0'])
			if 'camera2' in depth_images:
				self._frames_cam2_depth.append(depth_images['camera2'])
			if 'camera3' in depth_images:
				self._frames_cam3_depth.append(depth_images['camera3'])
			
		# 如果环境被重置，确保每个堆栈都填充相同的图像
		if reset:
			for _ in range(1, self.frame_stack):
				if 'rgb' in image_data:
					if 'camera1' in rgb_images:
						self._frames_cam1_rgb.append(rgb_images['camera1'])
					if 'camera0' in rgb_images:
						self._frames_cam0_rgb.append(rgb_images['camera0'])
					if 'camera2' in rgb_images:
						self._frames_cam2_rgb.append(rgb_images['camera2'])
					if 'camera3' in rgb_images:
						self._frames_cam3_rgb.append(rgb_images['camera3'])
				if 'depth' in image_data:
					if 'camera1' in depth_images:
						self._frames_cam1_depth.append(depth_images['camera1'])
					if 'camera0' in depth_images:
						self._frames_cam0_depth.append(depth_images['camera0'])
					if 'camera2' in depth_images:
						self._frames_cam2_depth.append(depth_images['camera2'])
					if 'camera3' in depth_images:
						self._frames_cam3_depth.append(depth_images['camera3'])	
		if 'rgb' in image_data:
			if 'camera1' in rgb_images:
				assert len(self._frames_cam1_rgb) == self.frame_stack
			if 'camera0' in rgb_images:
				assert len(self._frames_cam0_rgb) == self.frame_stack
			if 'camera2' in rgb_images:
				assert len(self._frames_cam2_rgb) == self.frame_stack
			if 'camera3' in rgb_images:
				assert len(self._frames_cam3_rgb) == self.frame_stack
		if 'depth' in image_data:
			if 'camera1' in depth_images:
				assert len(self._frames_cam1_depth) == self.frame_stack
			if 'camera0' in depth_images:
				assert len(self._frames_cam0_depth) == self.frame_stack
			if 'camera2' in depth_images:
				assert len(self._frames_cam2_depth) == self.frame_stack
			if 'camera3' in depth_images:
				assert len(self._frames_cam3_depth) == self.frame_stack

	def transform_obs(self, obs, reset=False):
		if self.render_mode == 'human':
			return obs['observation']
		if self.obs_mode == 'state':
			return obs['observation']
		elif self.obs_mode in ['rgb', 'depth','all']:
			self._update_frames(reset=reset)
			img_data={}
			camera_modes = self.mode2camera.get(self.xarm_camera_mode, [])
			if self.obs_mode in ['rgb', 'all']:
				img_data['rgb'] = {}
				for camera in camera_modes:
					if camera == 'camera0':
						img_data['rgb']['camera0'] = np.concatenate(list(self._frames_cam0_rgb), axis=-1 if self.channel_last else 0)
					elif camera == 'camera1':
						img_data['rgb']['camera1'] = np.concatenate(list(self._frames_cam1_rgb), axis=-1 if self.channel_last else 0)
					elif camera == 'camera2':
						img_data['rgb']['camera2'] = np.concatenate(list(self._frames_cam2_rgb), axis=-1 if self.channel_last else 0)
					else:
						img_data['rgb']['camera3'] = np.concatenate(list(self._frames_cam3_rgb), axis=-1 if self.channel_last else 0)
			if self.obs_mode in ['depth', 'all']:
				img_data['depth'] = {}
				for camera in camera_modes:
					if camera == 'camera0':
						img_data['depth']['camera0'] = np.concatenate(list(self._frames_cam0_depth), axis=-1 if self.channel_last else 0)
					elif camera == 'camera1':
						img_data['depth']['camera1'] = np.concatenate(list(self._frames_cam1_depth), axis=-1 if self.channel_last else 0)
					elif camera == 'camera2':
						img_data['depth']['camera2'] = np.concatenate(list(self._frames_cam2_depth), axis=-1 if self.channel_last else 0)
					else:
						img_data['depth']['camera3'] = np.concatenate(list(self._frames_cam3_depth), axis=-1 if self.channel_last else 0)
			return OrderedDict((('img', img_data), ('state', self.robot_state)))
		else:
			raise ValueError(f'Unknown obs_mode {self.obs_mode}. Can not be transformed.')

	def reset(self):
		return self.transform_obs(self._env.reset(), reset=True)

	def step(self, action):
		obs, reward, done, info = self._env.step(action)
		transformed_obs = self.transform_obs(obs)
		return transformed_obs, reward, done, info
