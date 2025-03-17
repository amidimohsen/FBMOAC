# In the name of ALLAH

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, Gamma, Normal, LogNormal
import torch.nn as nn
import numpy as np
# from utils import Buffer  # Assuming Buffer is defined below

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, CacheCapacity):
        super().__init__()
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.CacheCapacity = CacheCapacity
        
        # actor
        self.actor_dirichlet = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, self.action_dim // 2),
            nn.Softplus()
        )
        
        self.actor_Gamma1 = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )
        
        self.actor_Gamma2 = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, self.action_dim // 2),
            nn.Softplus()
        )
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, state):
        dirichlet_concentrate = self.actor_dirichlet(state)
        dirichlet_dist = Dirichlet(dirichlet_concentrate)
        
        gamma_params1 = self.actor_Gamma1(state)
        gamma_params2 = self.actor_Gamma2(state)
        Gamma_dist = LogNormal(gamma_params1, gamma_params2)
        
        state_value = self.critic(state)
        return dirichlet_dist, Gamma_dist, state_value

    def evaluate(self, state, action_cache, action_BW):
        dirichlet_dist, Gamma_dist, state_value = self.forward(state)
        
        logProb_cache = dirichlet_dist.log_prob(action_cache / self.CacheCapacity)
        logProb_BW = Gamma_dist.log_prob(action_BW).sum(dim=-1)
        
        entropy = dirichlet_dist.entropy() + 0.1 * Gamma_dist.entropy().sum(dim=-1)
        return logProb_cache, logProb_BW, state_value, entropy


class PPOAgent(object):
    def __init__(self, actor_critic, gamma, eps_clip, lr_actor_critic, K_epochs, Cache_capacity, device):
        self.train_device = device  # set device (CUDA or CPU)
        self.actor_critic = actor_critic.to(self.train_device)
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr_actor_critic)
        self.buffer = Buffer(self.train_device)  # Ensure buffer stores tensors on the correct device
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.timestep = 0
        self.Cache_capacity = Cache_capacity

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().to(self.train_device)
        dirichlet_dist, Gamma_dist, state_value = self.actor_critic.forward(state_tensor)
        
        action_cache = self.Cache_capacity * dirichlet_dist.sample()
        action_BW = Gamma_dist.sample()
        
        logProb_cache = dirichlet_dist.log_prob(action_cache / self.Cache_capacity)
        logProb_BW = Gamma_dist.log_prob(action_BW).sum(dim=-1)
        
        self.buffer.states.append(state_tensor)
        self.buffer.actions_cache.append(action_cache)
        self.buffer.actions_BW.append(action_BW)
        self.buffer.logprobs_cache.append(logProb_cache)
        self.buffer.logprobs_BW.append(logProb_BW)
        self.buffer.state_values.append(state_value)
        
        self.timestep += 1
        
        return action_cache.detach().cpu().numpy(), action_BW.detach().cpu().numpy()

    def update(self):
        states = torch.stack(self.buffer.states).to(self.train_device)
        actions_cache = torch.stack(self.buffer.actions_cache).to(self.train_device)
        actions_BW = torch.stack(self.buffer.actions_BW).to(self.train_device)
        old_logprobs_cache = torch.stack(self.buffer.logprobs_cache).detach().to(self.train_device)
        old_logprobs_BW = torch.stack(self.buffer.logprobs_BW).detach().to(self.train_device)
        old_state_values = torch.stack(self.buffer.state_values).detach().to(self.train_device)
        
        rewards = self.buffer.compute_discounted_rewards(self.gamma)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.train_device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-20)
        
        advantages = rewards - old_state_values.detach().squeeze()
        
        for _ in range(self.K_epochs):
            new_logProb_cache, new_logProb_BW, new_state_values, entropy = self.actor_critic.evaluate(states, actions_cache, actions_BW)
            new_state_values = torch.squeeze(new_state_values)
            
            ratios_cache = torch.exp(new_logProb_cache - old_logprobs_cache)
            ratios_BW = torch.exp(new_logProb_BW - old_logprobs_BW)
            
            surr1_cache = ratios_cache * advantages
            surr2_cache = torch.clamp(ratios_cache, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr1_BW = ratios_BW * advantages
            surr2_BW = torch.clamp(ratios_BW, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1_cache, surr2_cache).mean() - 100*torch.min(surr1_BW, surr2_BW).mean() - 0.01 * entropy.mean()
            critic_loss = F.mse_loss(new_state_values, rewards)
            loss = actor_loss + 5 * critic_loss
            
            self.actor_critic_optimizer.zero_grad()
            loss.backward()
            self.actor_critic_optimizer.step()
        
        self.buffer.clear()
        self.timestep = 0


class Buffer(object):
    def __init__(self, device):
        self.device = device
        self.states = []
        self.state_values = []
        self.actions_cache = []
        self.actions_BW = []
        self.rewards = []
        self.logprobs_cache = []
        self.logprobs_BW = []

    def clear(self):
        self.states         = []
        self.state_values   = []
        self.actions_cache  = []
        self.actions_BW     = []
        self.rewards        = []
        self.logprobs_cache = []
        self.logprobs_BW    = []
        
        
    def compute_discounted_rewards(self, gamma):
        """
        Compute discounted rewards for the episode.
        Assume rewards is a list of scalars.
        """
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        return discounted_rewards
