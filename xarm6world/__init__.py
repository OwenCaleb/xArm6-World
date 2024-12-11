from gym import register
import gym
from xarm6world.env.wrapper import TimeLimit,ActionRepeatWrapper,ObsTransformWrapper
import os

XARM6WORLD_TASKS = {
    'lift': dict(
        env='xArm6WorldLift-v0',
        action_space='xyzw',
        episode_length=50,
        description='Lift a cube above a height threshold'
    ),
    'reach': dict(
        env='xArm6WorldReach-v0',
        action_space='xyzw',
        episode_length=50,
        description='Reach a target location with the end effector'
    ),
    'push': dict(
        env='xArm6WorldPush-v0',
        action_space='xyzw',
        episode_length=50,
        description='Push a cube to a target location'
    ),
}

def format_task_name(task_name):
    return task_name.replace('_', ' ').title().replace(' ', '')

# Register each task
for task_name, task_info in XARM6WORLD_TASKS.items():
    register(
        id='xarm6world/'+task_info['env'], 
        entry_point=f"xarm6world.env:{format_task_name(task_name)}", 
    )

def make_env(params):
    """
	Make xArm6-World environment.
	"""
    task = params.get('task', 'lift')
    max_episode_steps = params.get('max_episode_steps', 50)
    repeat = params.get('repeat', 2)  
    obs_mode = params.get('obs_mode', 'rgb')  
    obs_size = params.get('obs_size', 84)  
    render_size = params.get('render_size', 224)  
    frame_stack = params.get('frame_stack', 1)  
    channel_last = params.get('channel_last', False)  
    xarm_camera_mode = params.get('xarm_camera_mode', 'mode2')  
    action_control_mode = params.get('action_control_mode', 'simple')  
    render_mode = params.get('render_mode', 'rgb_array')  

    env = gym.make(
        'xarm6world/'+XARM6WORLD_TASKS[task]["env"], 
		action_control_mode=action_control_mode,
    )
    env = TimeLimit(env,max_episode_steps=max_episode_steps)
    env= ActionRepeatWrapper(env, action=XARM6WORLD_TASKS[task]['action_space'], repeat=repeat)
    env = ObsTransformWrapper(env,obs_mode=obs_mode,obs_size=obs_size,frame_stack=frame_stack,channel_last=channel_last,xarm_camera_mode=xarm_camera_mode,render_mode=render_mode,render_size=render_size)
    return env
