
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
        
        # if the episode ended with termination compute the 
        # value for the next state
        if terminated and not truncated:
            with torch.no_grad():
                final = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, boot = policy.act(final)[2], None
            rew_buf += cfg.gamma * float(_)
            
        # reset the env if the epi is terminated or truncated
        if terminated or truncated:
            episode_returns.append(running)
            running = 0.0
            next_obs, _ = env.reset()
            
        obs = next_obs
        
    with torch.no_grad():
        last = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_value = policy.act(final)[2].item()
        
    batch = {
        "Obs": obs_buf, "Action": act_buf, "logp": logp_buf,
        "rewards": rew_buf, "values": val_buf, "dones": done_buf,
        "last_value": last_value
    }
    
    return batch, obs, episode_returns