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
    
class CategoricalPolicy(ActorCritic):
    """Discrete Actions"""
    
    is_discrete = True
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.policy_net = mlp(obs_dim, hidden_sizes, n_actions)
        self.value_net = mlp(obs_dim, hidden_sizes, 1)
        self.action_shape = ()
        self.action_dtype = np.int64
        
    def _dist(self, obs):
        return Categorical(logits=self.policy_net(obs)) #retuen categorical probs
    
    def act(self, obs):
        dist = self._dist(obs)
        action = dist.sample()
        
        return action, dist.log_prob(action), self.value_net(obs).squeeze(-1)
    
    def evaluate(self, obs, actions):
        dist = self._dist(obs)
        
        return dist.log_prob(actions), dist.entropy(), self.value_net(obs).squeeze(-1)
    
    def to_env_action(self, action):
        return int(action.item())
    
    
        