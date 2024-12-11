import xarm6world  
import cv2
from xarm6world import make_env

params = {
    'task': 'push',
    'max_episode_steps': 100,
    'repeat': 2,
    'obs_mode': 'rgb',
    'obs_size': 84,
    'render_size': 224,
    'frame_stack': 1,
    'channel_last': False,
    'xarm_camera_mode': 'mode2',
    'action_control_mode': 'simple',
    'render_mode': 'human'
}


env=make_env(params)

observation = env.reset()

for step in range(10000):  
    print(f"Step {step + 1}")

    action = env.action_space.sample()
    # action = np.clip(np.array([0.005, 0.005, -0.1, 1,0, 0, 0, -1.0], dtype=np.float32), -1.0, 1.0)

    observation, reward, done, info = env.step(action) 

    rendered_image = env.render(camera_name='camera0')
    
    if params['render_mode']=='rgb_array':
       rendered_image_bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
       cv2.imshow("Rendered Image", rendered_image_bgr)

    # print(f"Observation: {observation}")
    # print(f"Reward: {reward}")
    # print(f"Done: {done}")
    # print(f"Info: {info}")

    if done :
        print("Episode finished!")
        break

    if cv2.waitKey(42) & 0xFF == 27:
        print("Exited via ESC key!")
        break

env.close()
cv2.destroyAllWindows()