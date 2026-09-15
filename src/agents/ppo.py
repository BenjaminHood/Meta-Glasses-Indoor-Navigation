
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .config import PPOConfig
from .policies import make_policy

def collection_rollout(env, policy, cfg, obs, device):
    """Run the policy for the rollout steps and return a batch of transitions"""
    
    T = cfg.rollout_steps
    obs_buf = np.zeros((T, int(np.prod(env.observation_space.shape))), dtype=np.float32)
    act_buf = np.zero((T,) + policy.action_shape, dtype=policy.action_dtype)
    logp_buf = np.zeros(T, dtype=np.float32)
    rew_buf = np.zeros(T, dtype=np.float32)
    val_buf = np.zeros(T, dtype=np.float32)
    done_buf = np.zeros(T, dtype=np.float32)
    
    episode_returns = []
    running = 0.0
    
    for t in range(T):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logp, value = policy.act(obs_t)
            
        next_obs, reward, terminated, truncated, _ = env.step(policy.to_env_action(action))
        
        obs_buf[t] = obs
        act_buf[t] = action.squeeze(0).cpu().numpy()
        logp_buf[t] = logp.item()
        rew_buf[t] = reward
        val_buf[t] = value.item()
        done_buf[t] = float(terminated or truncated)
        running += reward
        
        