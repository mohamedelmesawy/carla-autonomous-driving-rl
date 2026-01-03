"""
Main Training Script for CARLA RL Agents
Supports DQN, SAC, and PPO algorithms with RGB or Semantic Segmentation inputs
"""

import argparse
import os
import cv2
import torch
import numpy as np
from datetime import datetime

from carla_env import CarlaGymEnv
from dqn_agent import DQNAgent
from sac_agent import SACAgent
from ppo_agent import PPOAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL agents in CARLA')

    # Algorithm selection
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['dqn', 'sac', 'ppo'],
                        help='RL algorithm to use (default: ppo)')

    # Environment settings
    parser.add_argument('--dim', type=int, default=84,
                        help='Image dimension for resize (default: 84)')
    parser.add_argument('--camera', type=str, default='seg',
                        choices=['rgb', 'seg'],
                        help='Camera type: rgb or seg (default: seg)')
    parser.add_argument('--town', type=str, default='Town01',
                        help='CARLA town name (default: Town01)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='CARLA server IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')

    # Training settings
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Show cv2 window (default: True)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable cv2 window')

    # Checkpoint settings
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Checkpoint save frequency in episodes (default: 50)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load checkpoint from file')

    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')

    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (uses algorithm default if not specified)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (uses algorithm default if not specified)')

    return parser.parse_args()


def setup_device(device_arg):
    """Setup the device for training."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    return device


def create_agent(algo, state_shape, num_actions, device, args):
    """Create the specified agent."""
    input_channels = 3 if args.camera == 'rgb' else 1

    if algo == 'dqn':
        # DQN-specific defaults
        lr = args.lr if args.lr is not None else 1e-4
        batch_size = args.batch_size if args.batch_size is not None else 32

        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            device=device,
            learning_rate=lr,
            gamma=args.gamma,
            batch_size=batch_size,
            buffer_size=100000,
            target_update_frequency=1000,
            use_dueling=True,
            image_dim=args.dim
        )

    elif algo == 'sac':
        # SAC-specific defaults
        lr = args.lr if args.lr is not None else 3e-4
        batch_size = args.batch_size if args.batch_size is not None else 64

        agent = SACAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            device=device,
            learning_rate=lr,
            gamma=args.gamma,
            batch_size=batch_size,
            buffer_size=100000,
            tau=0.005,
            image_dim=args.dim
        )

    elif algo == 'ppo':
        # PPO-specific defaults
        lr = args.lr if args.lr is not None else 3e-4
        batch_size = args.batch_size if args.batch_size is not None else 64

        agent = PPOAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            device=device,
            learning_rate=lr,
            gamma=args.gamma,
            batch_size=batch_size,
            rollout_size=2048,
            ppo_epochs=10,
            image_dim=args.dim
        )

    return agent


def train_dqn(env, agent, args, save_dir):
    """Train DQN agent."""
    print("Training DQN agent...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train_step()

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done or truncated:
                break

        # Update episode counter for epsilon decay
        agent.end_episode()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Print progress
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1}/{args.episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Length: {episode_length} | "
              f"Epsilon: {agent.get_epsilon():.3f}")

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'dqn_episode_{episode + 1}.pth')
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = os.path.join(save_dir, 'dqn_final.pth')
    agent.save_checkpoint(final_checkpoint)

    return episode_rewards, episode_lengths


def train_sac(env, agent, args, save_dir):
    """Train SAC agent."""
    print("Training SAC agent...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            train_info = agent.train_step()

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Print progress
        avg_reward = np.mean(episode_rewards[-100:])
        alpha = agent.get_alpha()
        print(f"Episode {episode + 1}/{args.episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Length: {episode_length} | "
              f"Alpha: {alpha:.4f}")

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'sac_episode_{episode + 1}.pth')
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = os.path.join(save_dir, 'sac_final.pth')
    agent.save_checkpoint(final_checkpoint)

    return episode_rewards, episode_lengths


def train_ppo(env, agent, args, save_dir):
    """Train PPO agent."""
    print("Training PPO agent...")

    episode_rewards = []
    episode_lengths = []
    current_state = env.reset()
    episode_reward = 0
    episode_length = 0

    for episode in range(args.episodes):
        while True:
            # Select action
            action, log_prob, value = agent.select_action(current_state)
            next_state, reward, done, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(current_state, action, reward, next_state,
                                   done or truncated, log_prob, value)

            episode_reward += reward
            episode_length += 1
            current_state = next_state

            # Train if rollout is complete
            if agent.is_ready():
                train_info = agent.train_step(next_state)
                if train_info:
                    print(f"  PPO Update - Policy Loss: {train_info['policy_loss']:.4f}, "
                          f"Value Loss: {train_info['value_loss']:.4f}, "
                          f"Entropy: {train_info['entropy']:.4f}")

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Print progress
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                print(f"Episode {episode + 1}/{args.episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Length: {episode_length}")

                # Reset for next episode
                episode_reward = 0
                episode_length = 0
                current_state = env.reset()
                break

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'ppo_episode_{episode + 1}.pth')
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = os.path.join(save_dir, 'ppo_final.pth')
    agent.save_checkpoint(final_checkpoint)

    return episode_rewards, episode_lengths


def main():
    """Main training function."""
    args = parse_args()

    # Handle render flag
    render_mode = args.render and not args.no_render

    # Setup device
    device = setup_device(args.device)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Create environment
    print("Creating CARLA environment...")
    env = CarlaGymEnv(
        host=args.host,
        port=args.port,
        image_dim=args.dim,
        camera_type=args.camera,
        town=args.town,
        render_mode=render_mode
    )

    # Get environment info
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}")

    # Create agent
    print(f"Creating {args.algo.upper()} agent...")
    agent = create_agent(args.algo, state_shape, num_actions, device, args)

    # Load checkpoint if specified
    if args.load is not None:
        print(f"Loading checkpoint from: {args.load}")
        agent.load_checkpoint(args.load)

    # Train agent
    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Camera: {args.camera}")
    print(f"Image dimension: {args.dim}")
    print("-" * 60)

    if args.algo == 'dqn':
        rewards, lengths = train_dqn(env, agent, args, args.save_dir)
    elif args.algo == 'sac':
        rewards, lengths = train_sac(env, agent, args, args.save_dir)
    elif args.algo == 'ppo':
        rewards, lengths = train_ppo(env, agent, args, args.save_dir)

    # Close environment
    env.close()

    print("\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Checkpoints saved in: {args.save_dir}")


if __name__ == '__main__':
    main()
    