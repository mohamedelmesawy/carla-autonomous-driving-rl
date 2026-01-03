"""
Proximal Policy Optimization (PPO) Agent for CARLA
Implements PPO with clipped objective and Generalized Advantage Estimation (GAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from networks import PPOActorNetwork, PPOCriticNetwork


class RolloutBuffer:
    """
    Rollout buffer for storing on-policy trajectories.
    """

    def __init__(self, capacity=2048):
        """
        Initialize the rollout buffer.

        Args:
            capacity: Maximum number of steps to store
        """
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def push(self, state, action, reward, next_state, done, log_prob, value):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            log_prob: Log probability of the action
            value: Value estimate for the state
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get_batches(self, batch_size, advantages, returns):
        """
        Generator that yields batches from the buffer.

        Args:
            batch_size: Size of each batch
            advantages: Computed advantages for each transition
            returns: Computed returns for each transition

        Yields:
            Batches of transitions
        """
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)

        for start in range(0, len(self.states), batch_size):
            end = min(start + batch_size, len(self.states))
            batch_indices = indices[start:end]

            yield (
                np.array(self.states)[batch_indices],
                np.array(self.actions)[batch_indices],
                np.array(self.log_probs)[batch_indices],
                np.array(self.values)[batch_indices],
                advantages[batch_indices],
                returns[batch_indices]
            )

    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    Uses clipped objective and GAE for advantage estimation.
    """

    def __init__(self, state_shape, num_actions, device='cpu',
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, rollout_size=2048, batch_size=64,
                 ppo_epochs=10, image_dim=84):
        """
        Initialize the PPO agent.

        Args:
            state_shape: Shape of the state observation
            num_actions: Number of discrete actions
            device: Device to run computations on
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            rollout_size: Number of steps to collect per rollout
            batch_size: Batch size for training
            ppo_epochs: Number of PPO epochs per rollout
            image_dim: Size of input image (assumed square)
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_size = rollout_size
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        # Determine input channels from state shape
        if len(state_shape) == 3:
            input_channels = state_shape[2]
        else:
            input_channels = 1

        # Initialize actor and critic networks
        self.actor = PPOActorNetwork(
            input_channels=input_channels,
            num_actions=num_actions,
            input_size=image_dim
        ).to(device)

        self.critic = PPOCriticNetwork(
            input_channels=input_channels,
            input_size=image_dim
        ).to(device)

        # Optimizer (shared for actor and critic)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(capacity=rollout_size)

        # Training statistics
        self.training_step = 0
        self.episode_step = 0

    def select_action(self, state):
        """
        Select an action using the current policy.

        Args:
            state: Current state observation

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Prepare state tensor
        if len(state.shape) == 3:
            state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Get action distribution
        with torch.no_grad():
            dist = self.actor.get_action_distribution(state_tensor)
            value = self.critic(state_tensor)

            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """
        Store a transition in the rollout buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            log_prob: Log probability of the action
            value: Value estimate for the state
        """
        self.rollout_buffer.push(state, action, reward, next_state, done, log_prob, value)
        self.episode_step += 1

    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0

        values = values + [next_value]
        dones = dones + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        returns = advantages + np.array(values[:-1])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(self, next_state):
        """
        Perform PPO training after collecting a rollout.

        Args:
            next_state: The state following the last transition in the rollout

        Returns:
            Dictionary of loss values if training occurred, None otherwise
        """
        if len(self.rollout_buffer) < self.batch_size:
            return None

        # Get value estimate for next state
        if len(next_state.shape) == 3:
            next_state_tensor = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        else:
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            next_value = self.critic(next_state_tensor).item()

        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(
            self.rollout_buffer.rewards,
            self.rollout_buffer.values,
            self.rollout_buffer.dones,
            next_value
        )

        # Convert buffer data to arrays
        states = np.array(self.rollout_buffer.states)
        actions = np.array(self.rollout_buffer.actions)
        old_log_probs = np.array(self.rollout_buffer.log_probs)
        old_values = np.array(self.rollout_buffer.values)

        # Train for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_old_values, batch_advantages, batch_returns in \
                    self.rollout_buffer.get_batches(self.batch_size, advantages, returns):

                # Convert to tensors
                batch_states = torch.from_numpy(batch_states).permute(0, 3, 1, 2).float().to(self.device)
                batch_actions = torch.from_numpy(batch_actions).long().to(self.device)
                batch_old_log_probs = torch.from_numpy(batch_old_log_probs).float().to(self.device)
                batch_old_values = torch.from_numpy(batch_old_values).float().to(self.device)
                batch_advantages = torch.from_numpy(batch_advantages).float().to(self.device)
                batch_returns = torch.from_numpy(batch_returns).float().to(self.device)

                # Get current policy distribution and values
                dist = self.actor.get_action_distribution(batch_states)
                current_values = self.critic(batch_states)

                # Compute log probabilities
                current_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratio for PPO
                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                # PPO policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(current_values, batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Clear the rollout buffer
        self.rollout_buffer.clear()
        self.episode_step = 0
        self.training_step += 1

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

    def is_ready(self):
        """Check if rollout buffer is ready for training."""
        return len(self.rollout_buffer) >= self.rollout_size

    def get_buffer_size(self):
        """Return current rollout buffer size."""
        return len(self.rollout_buffer)
