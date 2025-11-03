"""
Test script with GUARANTEED visualization
This script opens a MuJoCo viewer window to watch the robot
"""
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robot_env import RobotArmGraspEnv

def test_with_visualization(
    model_path="robot_arm_final_model.zip",
    vec_normalize_path="vec_normalize_final.pkl",
    n_episodes=10,
    slow_motion=False,
    pause_between_episodes=True
):
    """
    Test the robot with full visualization
    
    Args:
        model_path: Path to trained model
        vec_normalize_path: Path to normalization parameters
        n_episodes: Number of episodes to visualize
        slow_motion: If True, slows down the simulation for better viewing
        pause_between_episodes: If True, waits for user input between episodes
    """
    
    print("\n" + "="*70)
    print("ROBOT ARM VISUALIZATION TEST")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Slow motion: {slow_motion}")
    print("="*70 + "\n")
    
    # Create environment with human rendering
    print("Creating environment with visualization...")
    env = RobotArmGraspEnv(render_mode="human")
    
    # Wrap in DummyVecEnv for compatibility with VecNormalize
    vec_env = DummyVecEnv([lambda: env])
    
    # Load normalization parameters
    try:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("✓ Loaded normalization parameters\n")
    except FileNotFoundError:
        print(f"⚠ Warning: {vec_normalize_path} not found")
        print("  Testing without normalization...\n")
    
    # Load model
    print("Loading trained model...")
    model = PPO.load(model_path)
    print("✓ Model loaded\n")
    
    print("Starting visualization...")
    print("Watch the MuJoCo viewer window!\n")
    
    if slow_motion:
        print("(Running in SLOW MOTION for better viewing)\n")
    
    # Statistics tracking
    success_count = 0
    episode_stats = []
    
    try:
        for episode in range(n_episodes):
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1}/{n_episodes}")
            print(f"{'='*70}")
            
            obs = vec_env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done:
                # Get action from model (deterministic = best action)
                action, _states = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                # Render the environment
                env.render()
                
                # Optional: slow motion for better viewing
                if slow_motion:
                    time.sleep(0.02)  # 20ms delay per step
                
                # Print step info every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: Reward so far = {episode_reward:.2f}")
                
                if done:
                    # Get final statistics
                    cube_height = info[0].get("cube_height", 0)
                    success = cube_height >= 0.6
                    
                    if success:
                        success_count += 1
                        status = "✓ SUCCESS!"
                        emoji = "🎉"
                    else:
                        status = "✗ Failed"
                        emoji = "😞"
                    
                    episode_stats.append({
                        'episode': episode + 1,
                        'success': success,
                        'height': cube_height,
                        'reward': episode_reward,
                        'steps': step_count
                    })
                    
                    print(f"\n{emoji} {status}")
                    print(f"  Cube height: {cube_height:.3f}m")
                    print(f"  Total reward: {episode_reward:.2f}")
                    print(f"  Steps taken: {step_count}")
                    
                    # Pause between episodes if requested
                    if pause_between_episodes and episode < n_episodes - 1:
                        input("\nPress ENTER to continue to next episode...")
                    
                    break
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        success_rate = success_count / n_episodes
        avg_reward = np.mean([s['reward'] for s in episode_stats])
        avg_steps = np.mean([s['steps'] for s in episode_stats])
        avg_height = np.mean([s['height'] for s in episode_stats])
        
        print(f"\nSuccess Rate: {success_count}/{n_episodes} ({100*success_rate:.1f}%)")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Height: {avg_height:.3f}m")
        
        print("\nDetailed Episode Results:")
        print("-"*70)
        print(f"{'Ep':<4} {'Status':<10} {'Height':<10} {'Reward':<12} {'Steps':<8}")
        print("-"*70)
        for stat in episode_stats:
            status = "SUCCESS" if stat['success'] else "Failed"
            print(f"{stat['episode']:<4} {status:<10} {stat['height']:<10.3f} "
                  f"{stat['reward']:<12.2f} {stat['steps']:<8}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        print("\nClosing environment...")
        vec_env.close()
        print("Done!")


def quick_test(model_path="robot_arm_final_model.zip"):
    """Quick test with just 3 episodes, normal speed"""
    test_with_visualization(
        model_path=model_path,
        n_episodes=3,
        slow_motion=False,
        pause_between_episodes=False
    )


def detailed_test(model_path="robot_arm_final_model.zip"):
    """Detailed test with slow motion and pauses"""
    test_with_visualization(
        model_path=model_path,
        n_episodes=5,
        slow_motion=True,
        pause_between_episodes=True
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            print("Running quick test (3 episodes, normal speed)...\n")
            quick_test()
        elif sys.argv[1] == "detailed":
            print("Running detailed test (5 episodes, slow motion)...\n")
            detailed_test()
        elif sys.argv[1] == "best":
            print("Testing best model...\n")
            test_with_visualization(
                model_path="./logs/best_model.zip",
                n_episodes=10,
                slow_motion=False,
                pause_between_episodes=False
            )
        else:
            print("Unknown command!")
            print("\nUsage:")
            print("  python test_with_visualization.py              # Full test (10 episodes)")
            print("  python test_with_visualization.py quick        # Quick test (3 episodes)")
            print("  python test_with_visualization.py detailed     # Slow motion with pauses")
            print("  python test_with_visualization.py best         # Test best model")
    else:
        # Default: full visualization test
        print("Running full visualization test...\n")
        test_with_visualization(
            model_path="robot_arm_final_model.zip",
            n_episodes=10,
            slow_motion=False,
            pause_between_episodes=False
        )