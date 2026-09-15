"""Actor Critic Network"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Categorical, Normal

def mlp(in_dim: int, hidden_sizes, out_dim: int) -> nn.Sequential:
    layers = []
    last = in_dim
    
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
        
    layers.append(nn.Linear(last, out_dim))
    
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Interface every policy must satisfy"""
    
    is_discrete: bool
    action_shape: tuple
    action_dtype: type
    
    def act(self, obs):
        raise NotImplementedError
    
    def evaluate(self, obs, actions):
        raise NotImplementedError
    
    def to_env_action(self, action: torch.Tensor):
        """Convert a sampled action tensor into env.step()"""
        raise NotImplementedError
    
    