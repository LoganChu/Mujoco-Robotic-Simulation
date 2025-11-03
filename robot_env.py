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
        
        # Cache sensor addresses for efficient access
        self._cache_sensor_addresses()
        
        # Action space: 6 actuators (4 arm joints + 2 gripper fingers)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space: joint positions (6) + gripper pos (3) + cube pos (3) + cube vel (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = 2000  # INCREASED from 500 to give robot more time
        self.current_step = 0
        
        # Goal parameters - Use relative height from initial position
        self.goal_height_increase = 0.20  # Need to lift cube 20cm above starting position
        self.initial_cube_pos = np.array([0.5, 0.0, 0.445])
        self.initial_cube_height = None  # Track actual starting height
        
    def _cache_sensor_addresses(self):
        """
        Cache sensor addresses for efficient repeated access.
        
        EXPLANATION: Where does sensordata come from?
        ============================================
        When you create MjData (self.data = mujoco.MjData(self.model)), 
        MuJoCo automatically allocates and manages several arrays, including:
        
        - self.data.sensordata: A 1D array containing ALL sensor readings
        - self.data.qpos: Joint positions
        - self.data.qvel: Joint velocities
        - self.data.ctrl: Control inputs
        - self.data.xpos: Body positions in world frame
        - etc.
        
        The sensordata array is a FLAT array containing all sensor values
        concatenated together. For example, if you have:
        
        Sensor 0 (joint position, 1 value)  -> sensordata[0]
        Sensor 1 (frame position, 3 values) -> sensordata[1], sensordata[2], sensordata[3]
        Sensor 2 (joint velocity, 1 value)  -> sensordata[4]
        
        To find where each sensor's data starts in this flat array, you use:
        model.sensor_adr[sensor_id] -> gives the starting index
        
        This is MORE EFFICIENT than looking up names every time!
        """
        
        # Get sensor IDs
        gripper_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_pos")
        cube_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_pos")
        
        # Get starting addresses in sensordata array
        self.gripper_pos_adr = self.model.sensor_adr[gripper_sensor_id]
        self.cube_pos_adr = self.model.sensor_adr[cube_sensor_id]
        
        # Also cache body IDs for direct access
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        print(f"Sensor addresses cached:")
        print(f"  Gripper pos sensor starts at: sensordata[{self.gripper_pos_adr}]")
        print(f"  Cube pos sensor starts at: sensordata[{self.cube_pos_adr}]")
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize cube position slightly on table
        if self.np_random is not None:
            offset = self.np_random.uniform(-0.15, 0.15, size=2)
        else:
            offset = np.random.uniform(-0.15, 0.15, size=2)
        
        self.data.qpos[self.model.body_jntadr[self.cube_body_id]:self.model.body_jntadr[self.cube_body_id]+3] = [
            self.initial_cube_pos[0] + offset[0],
            self.initial_cube_pos[1] + offset[1],
            self.initial_cube_pos[2]
        ]
        
        # Reset velocities
        self.data.qvel[:] = 0
        
        # Forward simulation to stabilize
        for _ in range(10):  # Let cube settle on table
            mujoco.mj_step(self.model, self.data)
        
        # FIXED: Record the actual initial cube height after settling using correct sensor access
        self.initial_cube_height = self.data.sensordata[self.cube_pos_adr + 2]  # Z component
        
        # Alternative method: Direct body access (also correct!)
        # self.initial_cube_height = self.data.xpos[self.cube_body_id][2]
        
        print(f"Initial cube height set to: {self.initial_cube_height:.3f}m")
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        """
        Get observation vector.
        
        NOTE: Using cached sensor addresses for efficiency!
        """
        # Get joint positions (6 joints)
        joint_pos = self.data.qpos[:6].copy()
        
        # FIXED: Get gripper position using correct sensor address
        gripper_pos = self.data.sensordata[self.gripper_pos_adr:self.gripper_pos_adr+3].copy()
        
        # FIXED: Get cube position using correct sensor address
        cube_pos = self.data.sensordata[self.cube_pos_adr:self.cube_pos_adr+3].copy()
        
        # Get cube velocity (using body DOF address)
        cube_vel = self.data.qvel[self.model.body_dofadr[self.cube_body_id]:self.model.body_dofadr[self.cube_body_id]+3].copy()
        
        # Concatenate all observations
        obs = np.concatenate([joint_pos, gripper_pos, cube_pos, cube_vel])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        FIXED: All sensor accesses now use correct addresses!
        """
        # FIXED: Use cached sensor addresses
        gripper_pos = self.data.sensordata[self.gripper_pos_adr:self.gripper_pos_adr+3]
        cube_pos = self.data.sensordata[self.cube_pos_adr:self.cube_pos_adr+3]
        
        distance = np.linalg.norm(gripper_pos - cube_pos)
        
        # Calculate how much the cube has been lifted from initial position
        height_increase = cube_pos[2] - self.initial_cube_height if self.initial_cube_height is not None else 0
        
        return {
            "distance_to_cube": distance,
            "cube_height": cube_pos[2],
            "height_increase": height_increase,
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
        
        # Check termination based on HEIGHT INCREASE, not absolute height
        # Success = cube lifted 20cm above its starting position
        terminated = info["height_increase"] >= self.goal_height_increase
        truncated = self.current_step >= self.max_steps

        
        if terminated:
            # FIXED: Use correct sensor address for success message
            cube_height = self.data.sensordata[self.cube_pos_adr + 2]
            print(f"Current height: {cube_height:.3f}m")
            print(f"Initial height: {self.initial_cube_height:.3f}m")
            print(f"Height increase: {info['height_increase']:.3f}m")
            reward += 100.0  # Bonus for success
            print(f"SUCCESS! Cube lifted {info['height_increase']:.3f}m above starting position")
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, info):
        # Distance reward: negative distance between gripper and cube
        distance_reward = -info["distance_to_cube"] * 2.0
        
        # Height reward: encourage lifting the cube from initial position
        height_reward = info["height_increase"] * 10.0
        
        # Penalty for cube falling off table (absolute position check)
        cube_pos = info["cube_pos"]
        if cube_pos[2] < 0.4:  # Below table height
            return -10.0
        
        # Small penalty for each timestep to encourage efficiency
        time_penalty = -0.01
        
        total_reward = distance_reward + height_reward + time_penalty
        
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


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing Robot Arm Environment")
    print("="*60)
    
    # Create environment
    env = RobotArmGraspEnv(render_mode="human")
    
    # Test sensor access
    print("\nTesting sensor access after initialization:")
    obs, info = env.reset()
    print(f"Initial cube height (from info): {info['cube_height']:.3f}m")
    print(f"Initial height increase: {info['height_increase']:.3f}m (should be ~0)")
    print(f"Distance to cube: {info['distance_to_cube']:.3f}m")
    
    # Verify sensor data is correct
    print("\nVerifying sensor data:")
    print(f"Gripper position: {info['gripper_pos']}")
    print(f"Cube position: {info['cube_pos']}")
    
    # Test with random policy
    print("\n" + "="*60)
    print("Running random policy test...")
    print("="*60)
    
    for step in range(1000):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # Print info every 50 steps
        if step % 50 == 0:
            print(f"Step {step}: Height increase: {info['height_increase']:.3f}m, Distance: {info['distance_to_cube']:.3f}m, Reward: {reward:.2f}")
        
        if terminated or truncated:
            if terminated:
                print(f"\nSUCCESS! Episode finished. Cube lifted {info['height_increase']:.3f}m")
            else:
                print(f"\nTIMEOUT. Cube height increase: {info['height_increase']:.3f}m")
            
            obs, info = env.reset()
            print(f"Reset. New initial cube height: {info['cube_height']:.3f}m\n")
    
    env.close()
    print("\nTest complete!")