import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class RobotArmGraspEnv(gym.Env):
    """Custom Environment for Robot Arm Grasping Task"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, xml_path="robot_arm.xml"):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Render mode
        self.render_mode = render_mode
        self.viewer = None
        
        # Action space: 6 actuators (4 arm joints + 2 gripper fingers)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space: joint positions (6) + gripper pos (3) + cube pos (3) + cube vel (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = 500
        self.current_step = 0
        
        # Goal parameters
        self.goal_height = 0.6  # Target height for cube
        self.initial_cube_pos = np.array([0.5, 0.0, 0.445])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize cube position slightly on table
        if self.np_random is not None:
            offset = self.np_random.uniform(-0.15, 0.15, size=2)
        else:
            offset = np.random.uniform(-0.15, 0.15, size=2)
        
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.data.qpos[self.model.body_jntadr[cube_id]:self.model.body_jntadr[cube_id]+3] = [
            self.initial_cube_pos[0] + offset[0],
            self.initial_cube_pos[1] + offset[1],
            self.initial_cube_pos[2]
        ]
        
        # Reset velocities
        self.data.qvel[:] = 0
        
        # Forward simulation to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        # Get joint positions (6 joints)
        joint_pos = self.data.qpos[:6].copy()
        
        # Get gripper position (3D)
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_pos")
        gripper_pos = self.data.sensordata[gripper_id:gripper_id+3].copy()
        
        # Get cube position (3D)
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_pos")
        cube_pos = self.data.sensordata[cube_id:cube_id+3].copy()
        
        # Get cube velocity (3D)
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_vel = self.data.qvel[self.model.body_dofadr[cube_body_id]:self.model.body_dofadr[cube_body_id]+3].copy()
        
        # Concatenate all observations
        obs = np.concatenate([joint_pos, gripper_pos, cube_pos, cube_vel])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_pos")
        gripper_pos = self.data.sensordata[gripper_id:gripper_id+3]
        
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_pos")
        cube_pos = self.data.sensordata[cube_id:cube_id+3]
        
        distance = np.linalg.norm(gripper_pos - cube_pos)
        
        return {
            "distance_to_cube": distance,
            "cube_height": cube_pos[2],
            "gripper_pos": gripper_pos.copy(),
            "cube_pos": cube_pos.copy()
        }
    
    def step(self, action):
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        # Calculate reward
        reward = self._compute_reward(info)
        
        # Check termination conditions
        terminated = info["cube_height"] >= self.goal_height
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            reward += 100.0  # Bonus for success
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, info):
        # Distance reward: negative distance between gripper and cube
        distance_reward = -info["distance_to_cube"] * 2.0
        
        # Height reward: encourage lifting the cube
        height_reward = (info["cube_height"] - self.initial_cube_pos[2]) * 10.0
        
        # Penalty for cube falling off table
        cube_pos = info["cube_pos"]
        if cube_pos[2] < 0.4:  # Below table height
            return -10.0
        
        total_reward = distance_reward + height_reward
        
        return total_reward
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # Render to offscreen buffer
            camera = mujoco.MjvCamera()
            option = mujoco.MjvOption()
            
            mujoco.mjv_defaultCamera(camera)
            mujoco.mjv_defaultOption(option)
            
            camera.distance = 2.0
            camera.azimuth = 45
            camera.elevation = -20
            
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera=camera, scene_option=option)
            
            return renderer.render()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Example usage
if __name__ == "__main__":
    # Create environment
    env = RobotArmGraspEnv(render_mode="human")
    
    # Test random policy
    obs, info = env.reset()
    
    for _ in range(1000):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished. Cube height: {info['cube_height']:.3f}")
            obs, info = env.reset()
    
    env.close()