"""
Neural Network Architectures for CARLA RL Agents
CNN-based architectures for processing visual inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNNFeatureExtractor(nn.Module):
    """
    Shared CNN feature extractor for all agents.
    Processes image inputs and extracts high-level features.
    """

    def __init__(self, input_channels=3, hidden_dim=512, input_size=84):
        """
        Initialize the CNN feature extractor.

        Args:
            input_channels: Number of input channels (3 for RGB, 1 for segmentation)
            hidden_dim: Size of the hidden feature vector
            input_size: Size of input image (assumed square)
        """
        super(CNNFeatureExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Conv Layer 3
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),

            # Flatten
            nn.Flatten()
        )

        # Calculate the size of flattened features dynamically
        # For 84x84 input:
        #   Conv1: (84 - 8) / 4 + 1 = 20
        #   Conv2: (20 - 4) / 2 + 1 = 9
        #   Conv3: (9 - 3) / 2 + 1 = 4
        #   Flattened: 4 * 4 * 64 = 1024
        self.feature_size = self._calculate_conv_output_size(input_size)
        self.fc = nn.Linear(self.feature_size, hidden_dim)

    def _calculate_conv_output_size(self, input_size):
        """Calculate the size of the flattened conv output."""
        size = input_size
        # Conv1: kernel=8, stride=4
        size = (size - 8) // 4 + 1
        # Conv2: kernel=4, stride=2
        size = (size - 4) // 2 + 1
        # Conv3: kernel=3, stride=2
        size = (size - 3) // 2 + 1
        # 64 channels at the end
        return size * size * 64

    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Flattened feature vector
        """
        x = x / 255.0  # Normalize to [0, 1]
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for DQN algorithm.
    Maps observations to Q-values for each action.
    """

    def __init__(self, input_channels=3, num_actions=4, hidden_dim=512, input_size=84):
        """
        Initialize the DQN network.

        Args:
            input_channels: Number of input channels
            num_actions: Number of discrete actions
            hidden_dim: Size of hidden layers
            input_size: Size of input image (assumed square)
        """
        super(DQNNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(input_channels, hidden_dim, input_size)

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass to get Q-values.

        Args:
            x: Input observation

        Returns:
            Q-values for each action
        """
        features = self.feature_extractor(x)
        q_values = self.q_head(features)
        return q_values


class SACActorNetwork(nn.Module):
    """
    Actor network for SAC algorithm.
    Outputs action probabilities using a policy distribution.
    """

    def __init__(self, input_channels=3, num_actions=4, hidden_dim=512, input_size=84):
        """
        Initialize the SAC actor network.

        Args:
            input_channels: Number of input channels
            num_actions: Number of discrete actions
            hidden_dim: Size of hidden layers
            input_size: Size of input image (assumed square)
        """
        super(SACActorNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(input_channels, hidden_dim, input_size)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass to get action logits.

        Args:
            x: Input observation

        Returns:
            Action logits (before softmax)
        """
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        return logits

    def get_action_probs(self, x):
        """Get action probabilities."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class SACCriticNetwork(nn.Module):
    """
    Critic network (Q-network) for SAC algorithm.
    Estimates Q-values for state-action pairs.
    """

    def __init__(self, input_channels=3, num_actions=4, hidden_dim=512, input_size=84):
        """
        Initialize the SAC critic network.

        Args:
            input_channels: Number of input channels
            num_actions: Number of discrete actions
            hidden_dim: Size of hidden layers
            input_size: Size of input image (assumed square)
        """
        super(SACCriticNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(input_channels, hidden_dim, input_size)

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass to get Q-values.

        Args:
            x: Input observation

        Returns:
            Q-values for each action
        """
        features = self.feature_extractor(x)
        q_values = self.q_head(features)
        return q_values


class PPOActorNetwork(nn.Module):
    """
    Actor network for PPO algorithm.
    Outputs action distribution for policy gradient methods.
    """

    def __init__(self, input_channels=3, num_actions=4, hidden_dim=512, input_size=84):
        """
        Initialize the PPO actor network.

        Args:
            input_channels: Number of input channels
            num_actions: Number of discrete actions
            hidden_dim: Size of hidden layers
            input_size: Size of input image (assumed square)
        """
        super(PPOActorNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(input_channels, hidden_dim, input_size)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass to get action logits.

        Args:
            x: Input observation

        Returns:
            Action logits
        """
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        return logits

    def get_action_distribution(self, x):
        """
        Get action distribution for sampling.

        Args:
            x: Input observation

        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(x)
        return Categorical(logits=logits)


class PPOCriticNetwork(nn.Module):
    """
    Critic network for PPO algorithm.
    Estimates state value function V(s).
    """

    def __init__(self, input_channels=3, hidden_dim=512, input_size=84):
        """
        Initialize the PPO critic network.

        Args:
            input_channels: Number of input channels
            hidden_dim: Size of hidden layers
            input_size: Size of input image (assumed square)
        """
        super(PPOCriticNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(input_channels, hidden_dim, input_size)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass to get state value.

        Args:
            x: Input observation

        Returns:
            State value V(s)
        """
        features = self.feature_extractor(x)
        value = self.value_head(features)
        return value.squeeze(-1)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture.
    Separates value and advantage streams for better learning.
    """

    def __init__(self, input_channels=3, num_actions=4, hidden_dim=512, input_size=84):
        """
        Initialize the Dueling DQN network.

        Args:
            input_channels: Number of input channels
            num_actions: Number of discrete actions
            hidden_dim: Size of hidden layers
            input_size: Size of input image (assumed square)
        """
        super(DuelingDQNNetwork, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(input_channels, hidden_dim, input_size)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        """
        Forward pass to get Q-values using dueling architecture.

        Args:
            x: Input observation

        Returns:
            Q-values for each action
        """
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
