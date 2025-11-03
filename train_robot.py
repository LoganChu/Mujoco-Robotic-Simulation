"""
Training script using Stable-Baselines3 PPO algorithm
Install: pip install stable-baselines3[extra]
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from robot_env import RobotArmGraspEnv


class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment during training
    """
    def __init__(self, render_freq=1, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        
    def _on_step(self) -> bool:
        # Render every render_freq steps
        if self.n_calls % self.render_freq == 0:
            # Get the first environment from the vectorized wrapper
            env = self.training_env.envs[0]
            # Call render on the actual environment
            env.render()
        return True


def train_robot_arm(visualize=False):
    """Train the robot arm using PPO algorithm
    
    Args:
        visualize: If True, renders a single environment (slower training)
                   If False, runs all environments headless (faster training)
    """
    
    if visualize:
        # OPTION 1: Single environment with visualization (slower but visible)
        print("Training with visualization (single environment - slower)")
        env = make_vec_env(
            lambda: RobotArmGraspEnv(render_mode="human"),
            n_envs=1  # Just 1 environment to allow rendering
        )
    else:
        # OPTION 2: Multiple environments without rendering (faster)
        print("Training without visualization (4 parallel environments - faster)")
        env = make_vec_env(
            lambda: RobotArmGraspEnv(render_mode=None),
            n_envs=4  # 4 parallel environments for speed
        )
    
    # Wrap with VecNormalize for better training stability
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create evaluation environment (no rendering during eval)
    eval_env = make_vec_env(
        lambda: RobotArmGraspEnv(render_mode=None),
        n_envs=1
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        training=False
    )
    
    # Callbacks
    callbacks = []
    
    # Add render callback if visualizing
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
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="robot_arm_model"
    )
    
    callbacks.append(eval_callback)
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
    if visualize:
        print("Watch the robot learn in real-time!")
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
    
    # Test for multiple episodes
    n_episodes = 10
    success_count = 0
    
    for episode in range(n_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
            # Render
            env.render()
            
            if done[0]:
                cube_height = info[0].get("cube_height", 0)
                if cube_height >= 0.6:
                    success_count += 1
                    print(f"Episode {episode + 1}: SUCCESS! Height: {cube_height:.3f}, Reward: {episode_reward:.2f}, Steps: {step_count}")
                else:
                    print(f"Episode {episode + 1}: Failed. Height: {cube_height:.3f}, Reward: {episode_reward:.2f}, Steps: {step_count}")
                break
    
    print(f"\nSuccess rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        print("Testing trained model...")
        test_trained_model()
    else:
        # Training mode - ask user if they want visualization
        print("=" * 60)
        print("Robot Arm Training")
        print("=" * 60)
        print("\nVisualization options:")
        print("  [1] Train with visualization (1 env, slower, you can watch)")
        print("  [2] Train without visualization (4 envs, 4x faster)")
        print()
        
        choice = input("Choose option (1 or 2, default=2): ").strip()
        visualize = (choice == "1")
        
        print("\nTraining robot arm...")
        model, env = train_robot_arm(visualize=visualize)
        
        # Ask if user wants to test
        response = input("\nTraining complete. Test the model? (y/n): ")
        if response.lower() == 'y':
            test_trained_model()