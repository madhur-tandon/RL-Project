import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, modification, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.modification = modification
        self.seed = random.seed(seed)
        if self.modification:
            self.priorities = np.ones((buffer_size,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        if self.modification:
            probs = np.power(self.priorities[:len(self.memory)], 0.6)
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            experiences = [self.memory[idx] for idx in indices]
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if self.modification:
            weights = np.power(len(self.memory) * probs[indices], -0.4)
            weights /= weights.max()
            weights = torch.from_numpy(weights).float().to(device)
            return (states, actions, rewards, next_states, dones, indices, weights)
        else:
            return (states, actions, rewards, next_states, dones)

    def update_priorities(self, indices, priorities):
        for idx in indices:
            self.priorities[idx] = priorities

    def __len__(self):
        return len(self.memory)