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
        
    
    