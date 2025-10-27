"""
Training script using Stable-Baselines3 PPO algorithm
Install: pip install stable-baselines3[extra]
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robot_env import RobotArmGraspEnv
import time


class RenderCallback(BaseCallback):
    """
    Callback for rendering during training
    Shows one environment in real-time as the agent learns
    """
    def __init__(self, render_freq=1, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_env = None
        
    def _on_training_start(self):
        # Create a separate environment just for rendering
        self.render_env = RobotArmGraspEnv(render_mode="human")
        print("Live visualization enabled! Watch the robot learn in real-time.")
        
    def _on_step(self) -> bool:
        # Render at specified frequency
        if self.n_calls % self.render_freq == 0 and self.render_env is not None:
            # Get current observation from training
            obs = self.training_env.get_attr('_get_obs', indices=[0])[0]()
            
            # Copy state to render environment
            self.render_env.data.qpos[:] = self.training_env.get_attr('data', indices=[0])[0].qpos
            self.render_env.data.qvel[:] = self.training_env.get_attr('data', indices=[0])[0].qvel
            
            # Render
            self.render_env.render()
            time.sleep(0.01)  # Small delay for smooth visualization
            
        return True
    
    def _on_training_end(self):
        if self.render_env is not None:
            self.render_env.close()


def train_robot_arm(visualize=False):
    """Train the robot arm using PPO algorithm
    
    Args:
        visualize: If True, shows live visualization of training
    """
    
    # Create vectorized environment (parallel training)
    if visualize:
        # Use DummyVecEnv for single environment when visualizing
        env = DummyVecEnv([lambda: RobotArmGraspEnv(render_mode=None)])
        print("Training with live visualization (single environment)")
    else:
        # Use multiple parallel environments for faster training
        env = make_vec_env(
            lambda: RobotArmGraspEnv(render_mode=None),
            n_envs=4
        )
        print("Training with 4 parallel environments (no visualization)")
    
    # Wrap with VecNormalize for better training stability
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        lambda: RobotArmGraspEnv(render_mode=None),
        n_envs=1
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False
    )
    
    # Callbacks
    callbacks = []
    
    # Add render callback if visualization is enabled
    if visualize:
        render_callback = RenderCallback(render_freq=1)
        callbacks.append(render_callback)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="robot_arm_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Starting training...")
    if not visualize:
        print("Monitor progress with: tensorboard --logdir ./tensorboard_logs/")
    
    # Train the agent
    model.learn(
        total_timesteps=2_000_000,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save("robot_arm_final_model")
    env.save("vec_normalize_final.pkl")
    
    print("Training complete!")
    print("Model saved as: robot_arm_final_model.zip")
    
    return model, env


def watch_training_episode():
    """
    Watch a single training episode in real-time
    Useful for debugging and understanding the environment
    """
    env = RobotArmGraspEnv(render_mode="human")
    
    print("Watching random policy (no learning)...")
    print("The robot will take random actions to explore the environment.")
    
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(500):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            env.render()
            time.sleep(0.01)  # Slow down for viewing
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                print(f"Cube height: {info['cube_height']:.3f}m")
                print(f"Distance to cube: {info['distance_to_cube']:.3f}m")
                print(f"Total reward: {episode_reward:.2f}")
                break
    
    env.close()


def test_trained_model(model_path="robot_arm_final_model", vec_normalize_path="vec_normalize_final.pkl"):
    """Test the trained model with visualization"""
    
    # Create environment with rendering
    env = RobotArmGraspEnv(render_mode="human")
    
    # Load normalization parameters
    vec_env = make_vec_env(lambda: env, n_envs=1)
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    # Load trained model
    model = PPO.load(model_path, env=vec_env)
    
    print("Testing trained model...")
    
    # Test for multiple episodes
    n_episodes = 10
    success_count = 0
    
    for episode in range(n_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n--- Test Episode {episode + 1} ---")
        
        while True:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
            # Render
            env.render()
            time.sleep(0.01)
            
            if done[0]:
                cube_height = info[0].get("cube_height", 0)
                if cube_height >= 0.6:
                    success_count += 1
                    print(f"✓ SUCCESS! Height: {cube_height:.3f}m, Reward: {episode_reward:.2f}, Steps: {step_count}")
                else:
                    print(f"✗ Failed. Height: {cube_height:.3f}m, Reward: {episode_reward:.2f}, Steps: {step_count}")
                break
    
    print(f"\n{'='*50}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"{'='*50}")
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test":
            # Test trained model
            print("Testing trained model...")
            test_trained_model()
            
        elif mode == "watch":
            # Watch random policy
            print("Watching random policy...")
            watch_training_episode()
            
        elif mode == "train-visual":
            # Train with visualization
            print("Training with live visualization...")
            model, env = train_robot_arm(visualize=True)
            
        elif mode == "train":
            # Train without visualization (faster)
            print("Training without visualization...")
            model, env = train_robot_arm(visualize=False)
        else:
            print("Unknown mode. Use: train, train-visual, watch, or test")
    else:
        # Default: ask user
        print("\nRobot Arm Training Options:")
        print("1. Train with live visualization (slower, see each step)")
        print("2. Train without visualization (faster, 4x parallel)")
        print("3. Watch random policy (see environment)")
        print("4. Test trained model")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == "1":
            model, env = train_robot_arm(visualize=True)
        elif choice == "2":
            model, env = train_robot_arm(visualize=False)
        elif choice == "3":
            watch_training_episode()
        elif choice == "4":
            test_trained_model()
        else:
            print("Invalid choice")