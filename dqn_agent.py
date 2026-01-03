"""
Deep Q-Network (DQN) Agent for CARLA
Implements DQN with replay buffer, target network, and double DQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from networks import DQNNetwork, DuelingDQNNetwork


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores and samples transitions for off-policy learning.
    """

    def __init__(self, capacity=100_000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions as separate tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent with Double DQN and Dueling Architecture.
    """

    def __init__(self, state_shape, num_actions, device='cpu',
                 learning_rate=1e-4, gamma=0.99, buffer_size=100000,
                 batch_size=32, target_update_frequency=1000,
                 use_dueling=True, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_episodes=80000, image_dim=84):
        """
        Initialize the DQN agent.

        Args:
            state_shape: Shape of the state observation
            num_actions: Number of discrete actions
            device: Device to run computations on
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Maximum replay buffer size
            batch_size: Batch size for training
            target_update_frequency: Steps between target network updates
            use_dueling: Whether to use dueling architecture
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon for epsilon-greedy
            epsilon_decay_episodes: Number of episodes over which to decay epsilon
            image_dim: Size of input image (assumed square)
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.update_step = 0

        # Epsilon-greedy parameters (linear decay)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.current_episode = 0

        # Determine input channels from state shape
        if len(state_shape) == 3:
            input_channels = state_shape[2]
        else:
            input_channels = 1

        # Initialize networks
        NetworkClass = DuelingDQNNetwork if use_dueling else DQNNetwork
        self.q_network = NetworkClass(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        self.target_network = NetworkClass(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Training statistics
        self.training_step = 0

    def get_epsilon(self):
        """
        Calculate current epsilon value using linear decay.

        Returns:
            Current epsilon value
        """
        if self.current_episode < self.epsilon_decay_episodes:
            # Linear decay from epsilon_start to epsilon_end
            progress = self.current_episode / self.epsilon_decay_episodes
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        else:
            epsilon = self.epsilon_end
        return epsilon

    def select_action(self, state, eval_mode=False):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation
            eval_mode: If True, use greedy policy (no exploration)

        Returns:
            Selected action
        """
        if not eval_mode and random.random() < self.get_epsilon():
            return random.randrange(self.num_actions)

        with torch.no_grad():
            # Prepare state tensor
            if len(state.shape) == 3:
                state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            else:
                state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(self.device)

            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.from_numpy(states).permute(0, 3, 1, 2).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).permute(0, 3, 1, 2).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)

            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)

            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self.update_step += 1
        if self.update_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_step += 1

        return loss.item()

    def end_episode(self):
        """
        Call this at the end of each episode to update episode counter.
        This is used for linear epsilon decay.
        """
        self.current_episode += 1

    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_episode': self.current_episode,
            'training_step': self.training_step,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_episodes': self.epsilon_decay_episodes
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_episode = checkpoint.get('current_episode', 0)
        self.training_step = checkpoint['training_step']
        # Load epsilon parameters if available (for backward compatibility)
        self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = checkpoint.get('epsilon_end', self.epsilon_end)
        self.epsilon_decay_episodes = checkpoint.get('epsilon_decay_episodes', self.epsilon_decay_episodes)

    def set_episode(self, episode):
        """
        Set the current episode number (useful for resuming training).

        Args:
            episode: Episode number to set
        """
        self.current_episode = episode
        