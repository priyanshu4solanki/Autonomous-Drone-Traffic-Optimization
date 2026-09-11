import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
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
