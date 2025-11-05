import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os

class PandaPickEnv(gym.Env):
    """Custom Environment for Panda arm to pick up a block"""
    
    def __init__(self, xml_path="franka_emika_panda/scene.xml", render_mode=None, render_every_n_steps=1):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.render_every_n_steps = render_every_n_steps
        self.step_count = 0
        
        # Get important indices
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.left_gripper_id = mujoco.mj_name2id(self.model,  mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_gripper_id = mujoco.mj_name2id(self.model,  mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        
        # Action space: 7 joint velocities + 1 gripper
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        # Observation space: joint positions (7) + joint velocities (7) + 
        # gripper pos (3) + block pos (3) + goal pos (3) + block to gripper (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        
        self.max_steps = 2500
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize block position
        #block_x = np.random.uniform(0.3, 0.6)
        #block_y = np.random.uniform(-0.2, 0.2)
        #self.data.qpos[7:10] = [block_x, block_y, 0.025]  # block position
        #self.data.qpos[10:14] = [1, 0, 0, 0]  # block orientation (quaternion)
        
        # Set robot to home position
        self.data.qpos[0] = 0
        self.data.qpos[1] = 0
        self.data.qpos[2] = 0
        self.data.qpos[3] = -1.57079
        self.data.qpos[4] = 0
        self.data.qpos[5] = 1.57079
        self.data.qpos[6] = -0.7853
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        # Joint positions and velocities
        joint_pos = self.data.qpos[:7].copy()
        joint_vel = self.data.qvel[:7].copy()
        
        # Get end effector (gripper) position
        gripper_pos = self.data.xpos[self.model.nbody - 1][:3].copy()
        
        # Get block position
        block_pos = self.data.xpos[self.block_body_id][:3].copy()
        
        # Get goal position
        goal_pos = self.data.site_xpos[self.goal_site_id][:3].copy()
        
        # Relative vectors
        block_to_gripper = block_pos - gripper_pos
        
        obs = np.concatenate([
            joint_pos, joint_vel, gripper_pos, 
            block_pos, goal_pos, block_to_gripper
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        # Scale actions
        # Joint velocities (first 7 actions)
        joint_actions = action[:7] 

        # Gripper action (last action) - scale to gripper range
        gripper_action = (action[7] + 1) * 127.5  # Scale from [-1,1] to [0,255]
        
        # Apply actions
        self.data.ctrl[:7] = self.data.qpos[:7] + joint_actions * self.model.opt.timestep * 50
        self.data.ctrl[7] = gripper_action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Render if enabled
        if self.render_mode == "human":
            self.step_count += 1
            if self.step_count % self.render_every_n_steps == 0:
                self.render()
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        reward, info = self._compute_reward()
        
        self.current_step += 1
        terminated = info['success']
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self):
        
        gripper_pos = (self.data.xpos[self.left_gripper_id][:3] + self.data.xpos[self.right_gripper_id][:3])/2
        block_pos = self.data.xpos[self.block_body_id][:3]
        goal_pos = self.data.site_xpos[self.goal_site_id][:3]
        
        # Distance from gripper to block
        d_gripper_block = np.linalg.norm(gripper_pos - block_pos)

        # Distance from block to goal
        d_block_goal = np.linalg.norm(block_pos - goal_pos)

        # If you have block_size stored, use it. Otherwise estimate from geom
        block_size = 0.025
        
        # Calculate required gripper opening based on block size
        required_opening = block_size * 1.2  # 20% margin for alignment
        required_ctrl = (required_opening / 0.04) * 255  # Convert to control value
        
        # Check if gripper is closed around block
        gripper_ctrl = self.data.ctrl[7]  # Range: [0, 255], 0=closed, 255=open
        
        # Gripper should be: close enough to block AND closed enough for this block size
        is_grasped = (d_gripper_block < 0.05 and 
                    gripper_ctrl < required_ctrl * 1.5)  # Allow some tolerance
        
        # Phase-based rewards: reach -> grasp -> lift -> place
        reach_reward = 0
        grasp_reward = 0
        lift_reward = 0
        goal_reward = 0
        
        if not is_grasped:
            # Phase 1: Reach for the block
            reach_reward = -d_gripper_block * 3.0
            
            # Encourage closing gripper when near block
            if d_gripper_block < 0.05:
                # Reward for closing gripper to appropriate size for this object
                # Optimal control is around required_ctrl
                if gripper_ctrl > required_ctrl * 1.5:
                    # Too open - encourage closing
                    grasp_reward = -3.0 * (gripper_ctrl*1.5-required_ctrl)/255.0
                elif gripper_ctrl < required_ctrl * 0.5:
                    # Too closed - encourage opening
                    grasp_reward = 3.0 * (required_ctrl*1.5 - gripper_ctrl)/255.0
                else:
                    # Good gripper position
                    grasp_reward = 2.0
        else:
            # Phase 2: Block is grasped
            grasp_reward = 5.0  # Continuous bonus for maintaining grasp
            
            # Phase 3: Lift the block
            block_height = block_pos[2]
            lift_reward = (block_height - 0.025) * 3.0  # Reward for lifting off ground
            
            # Phase 4: Move to goal (only when grasped)
            goal_reward = -d_block_goal * 5.0
        
        # Success condition: block near goal
        success = d_block_goal < 0.05
        success_reward = 20.0 if success else 0.0
        
        # Total reward
        reward = reach_reward + grasp_reward + lift_reward + goal_reward + success_reward
        print("Overall Reward: ", reward)
        
        info = {
            'success': success,
            'd_gripper_block': d_gripper_block,
            'd_block_goal': d_block_goal,
            'block_height': block_pos[2],
            'is_grasped': is_grasped
        }
        
        return reward, info
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.viewer = None
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def train_panda(visualize=False, render_every_n_steps=10, continue_from=True):
    """Train the Panda arm using SAC algorithm
    
    Args:
        visualize: If True, show training in real-time
        render_every_n_steps: Render every N steps (to avoid slowdown)
    """
    
    # Create environment
    render_mode = "human" if visualize else None
    env = PandaPickEnv(
        xml_path="franka_emika_panda/scene.xml", 
        render_mode=render_mode,
        render_every_n_steps=render_every_n_steps
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./panda_models/',
        name_prefix='panda_sac'
    )
    
    # Create SAC model
    if continue_from:
        print(f"Loading model from {continue_from}...")
        model = SAC.load("panda_sac_final.zip", env=env)
        print("Continuing training from saved model...")
    else:
        # Create SAC model from scratch
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log="./panda_tensorboard/"
        )
    
    print("Starting training...")
    
    # Train the model
    model.learn(
        total_timesteps=50000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("panda_sac_final")
    print("Training completed!")
    
    return model


def test_trained_model(model_path="panda_sac_final.zip"):
    """Test the trained model with visualization"""
    
    env = PandaPickEnv(xml_path="franka_emika_panda/scene.xml", render_mode="human")
    model = SAC.load(model_path)
    
    print("Testing trained model...")
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            env.render()
            
            if terminated or truncated:
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {info['success']}")
                break
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        test_trained_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "train-viz":
        # Training with visualization
        print("Training with visualization (will be slower)...")
        render_every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        model = train_panda(visualize=True, render_every_n_steps=render_every, continue_from=True)
        
        # Test after training
        print("\nTesting trained model...")
        test_trained_model("panda_sac_final.zip")
    else:
        # Training mode without visualization (faster)
        print("Training without visualization (faster)...")
        print("Use 'python train_panda.py train-viz' to visualize during training")
        model = train_panda(visualize=False)
        
        # Test after training
        print("\nTesting trained model...")
        test_trained_model("panda_sac_final.zip")