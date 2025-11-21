import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer
import os
import torch
import multiprocessing

# Check GPU availability
if not torch.cuda.is_available():
    print("=" * 70)
    print("WARNING: GPU (CUDA) not available! Training will be MUCH slower.")
    print("=" * 70)
    USE_GPU = False
    device = "cpu"
else:
    USE_GPU = True
    device = "cuda"
    
    # GPU Optimization: Enable TF32 for Ampere+ GPUs 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # GPU Optimization: cuDNN benchmarking for optimal algorithms
    torch.backends.cudnn.benchmark = True
    
    # Detect GPU capabilities
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = gpu_props.total_memory / 1e9
    gpu_compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
    
    # Check for Tensor Cores (TF32 support) - Ampere+ has compute capability 8.0+
    has_tensor_cores = gpu_props.major >= 8
    
class PandaPickEnvGoalConditioned(gym.Env):
    """
    Goal-conditioned version of Panda environment for HER.
    Now learns to reach ANY goal, not just a fixed one!
    """
    
    def __init__(self, xml_path="franka_emika_panda/scene.xml", render_mode=None, curriculum_stage=1):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.curriculum_stage = curriculum_stage
        
        # Get important indices
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.left_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        # Action space: 7 joint velocities + 1 gripper
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        # Goal: [block_x, block_y, block_z, gripper_orient_x, y, z, gripper_x, gripper_y, gripper_z]
        self.goal_dim = 9  # Changed from 6 to 9
        
        # Observation space for GoalEnv
        # observation: robot state
        obs_dim = 23  # joint_pos(7) + joint_vel(7) + gripper_pos(3) + block_pos(3) + gripper_orient(3)
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
        })
        
        self.max_steps = 1500
        self.current_step = 0
        self.block_size = 0.025
    
    def set_curriculum_stage(self, stage):
        """Set the curriculum stage - callable from vectorized env"""
        self.curriculum_stage = stage
        if self.verbose:
            print(f"Environment curriculum stage set to {stage}")
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set robot to home position
        self.data.qpos[0] = 0
        self.data.qpos[1] = 0
        self.data.qpos[2] = 0
        self.data.qpos[3] = -1.57079
        self.data.qpos[4] = 0
        self.data.qpos[5] = 1.57079
        self.data.qpos[6] = -0.7853
        
        mujoco.mj_forward(self.model, self.data)
        
        # Sample a goal based on curriculum stage
        self.desired_goal = self._sample_goal()
        
        self.current_step = 0
        
        obs = self._get_obs()
        return obs, {}
    
    def _sample_goal(self):
        """Sample goals based on curriculum difficulty"""
        
        if self.curriculum_stage == 1:
            # Stage 1: Just get near the block (any position around it is fine)
            goal_pos = np.array([
                np.random.uniform(0.35, 0.45),  # Block area
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(0.025, 0.1)   # Low height OK
            ])
            # Don't care about orientation yet
            goal_orient = np.array([0, 0, -1])  # Placeholder
            
        elif self.curriculum_stage == 2:
            # Stage 2: Still near block, but now with orientation requirement
            # Block should still be on table (not lifted yet)
            goal_pos = np.array([
                np.random.uniform(0.35, 0.45),
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(0.025, 0.08)  # Keep block on table
            ])
            # NOW orientation matters - pointing down for grasping
            goal_orient = np.array([0, 0, -1])
            
        else:  # Stage 3
            # Stage 3: Full task - move block to target location
            goal_pos = self.data.site_xpos[self.goal_site_id][:3].copy()
            # Maintain correct orientation
            goal_orient = np.array([0, 0, -1])
        
        # Placeholder for desired gripper position (not used directly in reward)
        goal_gripper_pos = np.array([0, 0, 0])
        
        return np.concatenate([goal_pos, goal_orient, goal_gripper_pos]).astype(np.float32)
    
    def _get_obs(self):
        """Get observation in GoalEnv format"""
        # Joint positions and velocities
        joint_pos = self.data.qpos[:7].copy()
        joint_vel = self.data.qvel[:7].copy()
        
        # Gripper position (average of two fingers)
        gripper_pos = (self.data.xpos[self.left_gripper_id][:3] + 
                      self.data.xpos[self.right_gripper_id][:3]) / 2
        
        # Block position
        block_pos = self.data.xpos[self.block_body_id][:3].copy()
        
        # Gripper orientation (z-axis of hand)
        gripper_rot_id = self.model.body('hand').id
        gripper_rot = self.data.xmat[gripper_rot_id].reshape(3, 3)
        gripper_orient = gripper_rot[:, 2]  # Z-axis
        
        # Achieved goal: [block_pos(3), gripper_orient(3), gripper_pos(3)]
        achieved_goal = np.concatenate([
            block_pos,        # [0:3]
            gripper_orient,   # [3:6]
            gripper_pos       # [6:9]
        ]).astype(np.float32)
        
        # Robot state observation
        observation = np.concatenate([
            joint_pos, joint_vel, gripper_pos, block_pos, gripper_orient
        ]).astype(np.float32)
        
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': self.desired_goal.copy()
        }
    
    def step(self, action):
        # Apply action (same as before)
        print("step")
        joint_actions = action[:7]
        gripper_action = (action[7] + 1) * 127.5
        
        self.data.ctrl[:7] = self.data.qpos[:7] + joint_actions * self.model.opt.timestep * 50
        self.data.ctrl[7] = gripper_action
        
        mujoco.mj_step(self.model, self.data)
        
        if self.render_mode == "human":
            self.render()
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward based on goal achievement
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
        
        self.current_step += 1
        
        # Check success
        info = self._get_info(obs['achieved_goal'], obs['desired_goal'])
        terminated = info['is_success']
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute reward based on distance to goal.
        This is called by HER for relabeling!
        
        HER will call this with different desired_goals to relabel experiences.
        
        achieved_goal structure: [block_pos(3), gripper_orient(3), gripper_pos(3)]
        desired_goal structure: [target_block_pos(3), target_orient(3), placeholder_gripper(3)]
        """
        # Extract components from achieved_goal
        achieved_block_pos = achieved_goal[..., :3]
        achieved_gripper_orient = achieved_goal[..., 3:6]
        achieved_gripper_pos = achieved_goal[..., 6:9]
        
        # Extract components from desired_goal
        desired_block_pos = desired_goal[..., :3]
        desired_gripper_orient = desired_goal[..., 3:6]
        # Note: desired_goal[6:9] is placeholder, not used
        
        # COMPONENT 1: Gripper approaching block (critical early in learning!)
        gripper_to_block_dist = np.linalg.norm(achieved_gripper_pos - achieved_block_pos, axis=-1)
        print(gripper_to_block_dist)
        approach_reward = -gripper_to_block_dist * 20.0
        
        # COMPONENT 2: Block reaching target goal
        block_to_goal_dist = np.linalg.norm(achieved_block_pos - desired_block_pos, axis=-1)
        placement_reward = -block_to_goal_dist * 10.0
        
        # COMPONENT 3: Orientation alignment
        orient_similarity = np.sum(achieved_gripper_orient * desired_gripper_orient, axis=-1)
        orient_reward = (orient_similarity - 0.7) * 5.0
        
        # Curriculum-based weighting - FIXED ORDER!
        if self.curriculum_stage == 1:
            # Stage 1: ONLY focus on approaching block
            # Learn: "Move gripper close to block"
            reward = approach_reward * 1.0 + placement_reward * 0.0 + orient_reward * 0.0

            
        elif self.curriculum_stage == 2:
            # Stage 2: Approach + Orientation
            # Learn: "Get close to block WITH correct orientation for grasping"
            reward = approach_reward * 0.7 + placement_reward * 0.0 + orient_reward * 0.8
            
        else:  # Stage 3
            # Stage 3: All components - full task
            # Learn: "Grasp block with correct orientation AND move it to target"
            reward = approach_reward * 0.3 + placement_reward * 1.0 + orient_reward * 0.8
        
        return reward
    
    def _get_info(self, achieved_goal, desired_goal):
        """Check if goal is achieved"""
        achieved_pos = achieved_goal[:3]  # Block position
        desired_pos = desired_goal[:3]
        achieved_orient = achieved_goal[3:6]  # Gripper orientation
        desired_orient = desired_goal[3:6]
        achieved_gripper_pos = achieved_goal[6:9]  # Gripper position
        
        pos_dist = np.linalg.norm(achieved_pos - desired_pos)
        orient_similarity = np.dot(achieved_orient, desired_orient)
        gripper_to_block = np.linalg.norm(achieved_gripper_pos - achieved_pos)
        
        # Success criteria follows curriculum progression
        if self.curriculum_stage == 1:
            # Stage 1: Success = gripper is close to block (within 5cm)
            is_success = gripper_to_block < 0.2
            
        elif self.curriculum_stage == 2:
            # Stage 2: Success = gripper close to block WITH correct orientation
            is_success = (gripper_to_block < 0.1) and (orient_similarity > 0.85)
            
        else:  # Stage 3
            # Stage 3: Success = block moved to target location with orientation
            is_success = (pos_dist < 0.05) and (orient_similarity > 0.85)
        
        return {
            'is_success': is_success,
            'pos_dist': pos_dist,
            'orient_similarity': orient_similarity,
            'gripper_to_block': gripper_to_block
        }
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        env = PandaPickEnvGoalConditioned(
            xml_path="franka_emika_panda/scene.xml",
            render_mode=None,  # No rendering in parallel envs
            curriculum_stage=1
        )
        env.reset(seed=seed + rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """
    Callback to automatically progress through curriculum stages
    Works with both DummyVecEnv and SubprocVecEnv
    """
    
    def __init__(self, env, window_size=100, advance_threshold=0.8, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.window_size = window_size
        self.advance_threshold = advance_threshold
        self.success_history = []
        self.max_stage = 3
        self.current_stage = 1  # Track stage in callback instead
        
    def _on_step(self) -> bool:
        # Check all parallel environments for done episodes
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        
        # Handle multiple environments
        for i, done in enumerate(dones):
            if done:
                success = infos[i].get('is_success', False)
                self.success_history.append(float(success))
                
                # Keep only recent history
                if len(self.success_history) > self.window_size:
                    self.success_history.pop(0)
        
        # Check if should advance curriculum
        if len(self.success_history) >= self.window_size:
            success_rate = np.mean(self.success_history)
            
            if self.verbose >= 1:
                print(f"\nStage {self.current_stage} - Success Rate: {success_rate:.2%}")
            
            if success_rate > self.advance_threshold and self.current_stage < self.max_stage:
                # Advance curriculum
                self.current_stage += 1
                
                # Update curriculum stage in all environments
                # This works for both DummyVecEnv and SubprocVecEnv
                self.env.env_method('set_curriculum_stage', self.current_stage)
                
                self.success_history = []
                
                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"🎓 CURRICULUM ADVANCED TO STAGE {self.current_stage}!")
                    print(f"{'='*60}\n")
        
        return True


def train_with_her_and_curriculum(visualize=False, total_timesteps=500000, n_envs=8):
    """
    Train Panda with HER + Curriculum Learning
    No demonstrations needed!
    
    :param visualize: Show visualization (only works with n_envs=1)
    :param total_timesteps: Total training steps
    :param n_envs: Number of parallel environments (default: 8)
    """
    
    print("="*60)
    print("Training with HER + Curriculum Learning")
    if n_envs > 1:
        print(f"Using {n_envs} parallel environments for {n_envs}x speedup!")
    print("="*60)
    print("\nStage 1: Learn to approach block (position only)")
    print("Stage 2: Learn to approach with correct orientation (ready to grasp)")
    print("Stage 3: Full task - grasp and move to target\n")
    
    # Create multiple parallel environments
    if n_envs > 1 and not visualize:
        print(f"Creating {n_envs} parallel environments...")
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        # Single environment (for debugging or visualization)
        if visualize and n_envs > 1:
            print("Warning: Visualization only works with single environment. Setting n_envs=1")
            n_envs = 1
        
        render_mode = "human" if visualize else None
        env = PandaPickEnvGoalConditioned(
            xml_path="franka_emika_panda/scene.xml",
            render_mode=render_mode,
            curriculum_stage=1
        )
        env = DummyVecEnv([lambda: env])
    
    # Create curriculum callback
    curriculum_callback = CurriculumCallback(
        env=env,
        window_size=100,
        advance_threshold=0.8,
        verbose=1
    )
    
    # Create SAC model with HER
    model = SAC(
        "MultiInputPolicy",  # Required for GoalEnv (Dict observations)
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # Number of HER goals per transition
            goal_selection_strategy='future',  # Use "future" strategy
        ),
        verbose=1,
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=10500, 
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        device=device,  # Use GPU for neural network training
        tensorboard_log="./panda_her_tensorboard/"
    )
    
    print("\nStarting training with HER...")
    print("HER will automatically relabel failed attempts as successes for different goals!")
    print(f"Training device: {device}")
    if USE_GPU:
        print(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("panda_her_curriculum_final")
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Clean up
    env.close()
    
    return model


def test_her_model(model_path="panda_her_curriculum_final.zip", num_episodes=10):
    """Test the trained HER model"""
    
    print("\nTesting trained model...")
    
    env = PandaPickEnvGoalConditioned(
        xml_path="franka_emika_panda/scene.xml",
        render_mode="human",
        curriculum_stage=3  # Test on hardest stage
    )
    
    model = SAC.load(model_path, env=env)
    
    successes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Desired goal: {obs['desired_goal']}")
        
        for step in range(10000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            env.render()
            
            if terminated or truncated:
                success = info['is_success']
                successes += success
                print(f"Achieved goal: {obs['achieved_goal']}")
                print(f"Success: {success}, Reward: {episode_reward:.2f}")
                print(f"Position error: {info['pos_dist']:.4f}m")
                print(f"Orientation similarity: {info['orient_similarity']:.4f}")
                print(f"Gripper to block: {info['gripper_to_block']:.4f}m")
                break
    
    print(f"\n{'='*60}")
    print(f"Overall Success Rate: {successes}/{num_episodes} = {successes/num_episodes:.1%}")
    print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test trained model
        test_her_model()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "train-viz":
        # Train with visualization (slower, single env only)
        print("Training with visualization (single environment)...")
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        model = train_with_her_and_curriculum(
            visualize=True, 
            total_timesteps=timesteps,
            n_envs=1  # Must be 1 for visualization
        )
        
        print("\nTesting...")
        test_her_model()
        
    else:
        # Train without visualization (faster with parallel envs)
        print("Training without visualization (parallel environments)")
        print("Use 'python script.py train-viz' to visualize training")
        
        timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
        
        # Detect number of CPU cores
        n_cores = multiprocessing.cpu_count()
        n_envs = min(n_cores - 1, 16)  # Use cores-1, max 16 envs
        print(f"Detected {n_cores} CPU cores, using {n_envs} parallel environments")
        
        model = train_with_her_and_curriculum(
            visualize=False, 
            total_timesteps=timesteps,
            n_envs=n_envs
        )
        
        print("\nTesting...")
        test_her_model()


# Expected training timeline with parallel environments (8 envs):
# Episodes 0-2000: Stage 1 (approach block) - Success rate: 0% → 85%  (~10-15 minutes)
# Episodes 2000-6000: Stage 2 (approach + orient) - Success rate: 20% → 80%  (~20-30 minutes)
# Episodes 6000+: Stage 3 (grasp and place) - Success rate: 30% → 70%+  (~20-30 minutes)
#
# Total training time with 8 parallel envs:
#   - CPU only: ~30-60 minutes (vs 2-4 hours sequential)
#   - CPU + GPU: ~20-40 minutes (vs 30-60 min sequential)
#
# Speedup: 4-8x faster with parallel environments!
# Final success rate on full task: 65-75% (without any demonstrations!)