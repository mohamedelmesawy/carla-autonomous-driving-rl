"""
Soft Actor-Critic (SAC) Agent for CARLA (Discrete Action Version)
Implements SAC with discrete actions, twin Q-networks, and automated entropy tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from networks import SACActorNetwork, SACCriticNetwork


class ReplayBuffer:
    """
    Experience Replay Buffer for SAC.
    Stores and samples transitions for off-policy learning.
    """

    def __init__(self, capacity=100000):
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


class SACAgent:
    """
    Soft Actor-Critic Agent for Discrete Actions.
    Uses twin Q-networks and automated entropy tuning.
    """

    def __init__(self, state_shape, num_actions, device='cpu',
                 learning_rate=3e-4, gamma=0.99, buffer_size=100000,
                 batch_size=64, tau=0.005, alpha=0.2, auto_entropy_tuning=True,
                 target_entropy=None, image_dim=84):
        """
        Initialize the SAC agent.

        Args:
            state_shape: Shape of the state observation
            num_actions: Number of discrete actions
            device: Device to run computations on
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Maximum replay buffer size
            batch_size: Batch size for training
            tau: Soft update coefficient for target networks
            alpha: Entropy regularization coefficient
            auto_entropy_tuning: Whether to automatically tune alpha
            target_entropy: Target entropy for automatic tuning
            image_dim: Size of input image (assumed square)
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning

        # Determine input channels from state shape
        if len(state_shape) == 3:
            input_channels = state_shape[2]
        else:
            input_channels = 1

        # Initialize actor network
        self.actor = SACActorNetwork(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        # Initialize twin critic networks
        self.critic1 = SACCriticNetwork(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        self.critic2 = SACCriticNetwork(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        # Initialize target critic networks
        self.target_critic1 = SACCriticNetwork(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        self.target_critic2 = SACCriticNetwork(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Entropy coefficient
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -np.log(1.0 / num_actions) * 0.98  # Slightly lower than max entropy
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = 0.2
        else:
            self.alpha = alpha
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Training statistics
        self.training_step = 0

    def select_action(self, state, eval_mode=False):
        """
        Select an action using the current policy.

        Args:
            state: Current state observation
            eval_mode: If True, use deterministic policy (most probable action)

        Returns:
            Selected action
        """
        with torch.no_grad():
            # Prepare state tensor
            if len(state.shape) == 3:
                state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            else:
                state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # Get action probabilities
            action_probs = self.actor.get_action_probs(state_tensor)

            if eval_mode:
                # Deterministic action (most probable)
                action = action_probs.argmax(dim=1).item()
            else:
                # Stochastic action (sample from policy)
                dist = torch.distributions.Categorical(probs=action_probs)
                action = dist.sample().item()

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

    def _compute_target_q(self, rewards, next_states, dones):
        """
        Compute target Q-values using target critics.

        Args:
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Target Q-values
        """
        with torch.no_grad():
            # Get action probabilities for next states
            next_action_probs = self.actor.get_action_probs(next_states)

            # Compute log probabilities
            next_log_probs = torch.log(next_action_probs + 1e-8)

            # Compute target Q-values from both critics
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)

            # Take minimum Q-value
            next_q = torch.min(next_q1, next_q2)

            # Compute weighted Q-value
            next_q = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)

            # Compute target
            target_q = rewards + (1 - dones) * self.gamma * next_q

        return target_q

    def train_step(self):
        """
        Perform one training step.

        Returns:
            Dictionary of loss values if training occurred, None otherwise
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

        # Compute target Q-values
        target_q = self._compute_target_q(rewards, next_states, dones)

        # Train critics
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        self.critic2_optimizer.step()

        # Train actor
        action_probs = self.actor.get_action_probs(states)
        log_probs = torch.log(action_probs + 1e-8)

        with torch.no_grad():
            q1 = self.critic1(states)
            q2 = self.critic2(states)
            q = torch.min(q1, q2)

        # SAC actor loss: maximize E[Q(s, a) - alpha * log(pi(a|s))]
        actor_loss = (action_probs * (self.alpha * log_probs - q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = None
        if self.auto_entropy_tuning:
            # Compute current entropy
            entropy = -(action_probs * log_probs).sum(dim=1).mean()

            # Update alpha to match target entropy
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach())

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        self.training_step += 1

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else None,
            'alpha': self.alpha
        }

    def _soft_update(self, target_network, source_network):
        """
        Soft update target network parameters.

        Args:
            target_network: Target network to update
            source_network: Source network to copy from
        """
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'training_step': self.training_step,
            'alpha': self.alpha
        }

        if self.auto_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha.item()
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.alpha = checkpoint['alpha']

        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data.fill_(checkpoint['log_alpha'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

    def get_alpha(self):
        """Return current entropy coefficient."""
        return self.alpha

    def set_alpha(self, alpha):
        """
        Set entropy coefficient to a specific value.

        Args:
            alpha: New alpha value
        """
        self.alpha = alpha
        if self.auto_entropy_tuning:
            self.log_alpha.data.fill_(np.log(alpha))
