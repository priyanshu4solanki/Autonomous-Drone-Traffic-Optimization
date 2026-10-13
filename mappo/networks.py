import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    """Legacy MLP actor used to load checkpoints created before the CNN policy."""

    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def distribution(self, obs, action_mask=None):
        logits = self.net(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return Categorical(logits=logits)


class MapCNNActor(nn.Module):
    def __init__(self, obs_dim, action_dim, map_channels=6, hidden=128):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + 32 * 2 * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def distribution(self, obs, local_map, action_mask=None):
        map_features = self.map_encoder(local_map)
        logits = self.policy(torch.cat((obs, map_features), dim=-1))
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return Categorical(logits=logits)


class CentralCritic(nn.Module):
    def __init__(self, state_dim, num_agents, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_agents),
        )

    def forward(self, state):
        return self.net(state)
