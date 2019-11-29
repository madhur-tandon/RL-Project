import numpy as np
import random
import copy

from actor import Actor
from critic import Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from noise import OUNoise
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, modification, random_seed, fc1_units=400, fc2_units=300):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.modification = modification

        self.actor_local = Actor(state_size, action_size, modification, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, modification, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, modification, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, modification, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, modification, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        if self.modification:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        if self.modification:
            priorities = np.sqrt(critic_loss.detach().cpu().data.numpy())
            self.memory.update_priorities(indices, priorities)
            critic_loss = critic_loss * weights
            critic_loss = critic_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.modification:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        if self.modification:
            priorities = np.sqrt(actor_loss.detach().cpu().data.numpy())
            self.memory.update_priorities(indices, priorities)
            actor_loss = actor_loss * weights
            actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, path):
        torch.save(self.actor_local.state_dict(), 
                   path+'_actor.pth')
        torch.save(self.critic_local.state_dict(),
                   path+'_critic.pth')