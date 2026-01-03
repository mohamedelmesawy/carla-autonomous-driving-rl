"""
Inference Script for CARLA RL Agents
Loads a trained checkpoint and runs the agent in the environment
"""

import argparse
import os
import cv2
import torch
import numpy as np

from carla_env import CarlaGymEnv
from dqn_agent import DQNAgent
from sac_agent import SACAgent
from ppo_agent import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with trained RL agents in CARLA')

    # Checkpoint settings
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pth)')

    # Environment settings
    parser.add_argument('--camera', type=str, default='seg',
                        choices=['rgb', 'seg'],
                        help='Camera type: rgb or seg (default: seg)')
    parser.add_argument('--dim', type=int, default=84,
                        help='Image dimension for resize (default: 84)')
    parser.add_argument('--town', type=str, default='Town01',
                        help='CARLA town name (default: Town01)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='CARLA server IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')

    # Inference settings
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable cv2 window')
    parser.add_argument('--save_video', action='store_true',
                        help='Save video of the inference run')
    parser.add_argument('--video_path', type=str, default='inference_video.mp4',
                        help='Path to save video (default: inference_video.mp4)')

    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')

    return parser.parse_args()


def setup_device(device_arg):
    """Setup the device for inference."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    return device


def detect_algorithm_from_checkpoint(checkpoint_path):
    """
    Detect the algorithm type from the checkpoint filename.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Algorithm name ('dqn', 'sac', or 'ppo')
    """
    filename = os.path.basename(checkpoint_path).lower()

    if 'dqn' in filename:
        return 'dqn'
    elif 'sac' in filename:
        return 'sac'
    elif 'ppo' in filename:
        return 'ppo'
    else:
        # Try to detect from checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'actor_state_dict' in checkpoint:
            if 'log_alpha' in checkpoint:
                return 'sac'  # SAC has log_alpha
            else:
                return 'ppo'  # PPO doesn't have log_alpha
        elif 'q_network_state_dict' in checkpoint:
            return 'dqn'

        raise ValueError(f"Could not detect algorithm from checkpoint: {checkpoint_path}")


def create_agent(algo, state_shape, num_actions, device, checkpoint_path, image_dim):
    """
    Create agent and load checkpoint.

    Args:
        algo: Algorithm name ('dqn', 'sac', or 'ppo')
        state_shape: Shape of the state observation
        num_actions: Number of discrete actions
        device: Device to run on
        checkpoint_path: Path to checkpoint file
        image_dim: Size of input image (assumed square)

    Returns:
        Loaded agent
    """
    input_channels = 3 if len(state_shape) == 3 and state_shape[2] == 3 else 1

    if algo == 'dqn':
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            device=device,
            use_dueling=True,
            image_dim=image_dim
        )
    elif algo == 'sac':
        agent = SACAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            device=device,
            image_dim=image_dim
        )
    elif algo == 'ppo':
        agent = PPOAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            device=device,
            image_dim=image_dim
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent.load_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully!")

    return agent


def run_inference(env, agent, args, algo):
    """
    Run inference with the trained agent.

    Args:
        env: CARLA environment
        agent: Trained agent
        args: Command line arguments
        algo: Algorithm name
    """
    print(f"\nRunning inference for {args.episodes} episodes...")
    print("-" * 60)

    episode_rewards = []
    episode_lengths = []

    # Video writer setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.video_path, fourcc, 20.0,
                                       (args.dim, args.dim))

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action (eval mode = no exploration)
            if algo == 'ppo':
                action, _, _ = agent.select_action(state)
            else:
                action = agent.select_action(state, eval_mode=True)

            # Step environment
            next_state, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            state = next_state

            done = done or truncated

            # Save frame if recording video
            if video_writer is not None and env.camera_data is not None:
                frame = env.camera_data.copy()
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{args.episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Length: {episode_length} | "
              f"Collision: {info.get('collision', False)} | "
              f"Lane Invasion: {info.get('lane_invasion', False)}")

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved to: {args.video_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"Total episodes: {args.episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Worst episode reward: {np.min(episode_rewards):.2f}")


def main():
    """Main inference function."""
    args = parse_args()

    # Validate checkpoint file
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Setup device
    device = setup_device(args.device)

    # Detect algorithm from checkpoint
    algo = detect_algorithm_from_checkpoint(args.checkpoint)
    print(f"Detected algorithm: {algo.upper()}")

    # Create environment
    print("Creating CARLA environment...")
    env = CarlaGymEnv(
        host=args.host,
        port=args.port,
        image_dim=args.dim,
        camera_type=args.camera,
        town=args.town,
        render_mode=not args.no_render
    )

    # Get environment info
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}")

    # Create and load agent
    agent = create_agent(algo, state_shape, num_actions, device, args.checkpoint, args.dim)

    # Run inference
    run_inference(env, agent, args, algo)

    # Close environment
    env.close()
    print("\nInference completed!")


if __name__ == '__main__':
    main()
