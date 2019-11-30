import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import hidden_init

class Actor(nn.Module):
    def __init__(self, state_size, action_size, modification, seed, fc1_units=400, fc2_units=300, environment='pendulum'):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.modification = modification
        self.environment = environment
        if self.environment == 'pendulum':
            self.fc1 = nn.Linear(state_size, fc1_units)
            if self.modification:
                self.bn1 = nn.BatchNorm1d(fc1_units)
                self.bn2 = nn.BatchNorm1d(fc2_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.reset_parameters()
        else:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if self.environment == 'pendulum':
            if self.modification:
                y = self.fc1(state)
                if len(y.size())==1:
                    y = y.unsqueeze(0)
                y = self.bn1(y)
                y = y.squeeze(0)
                x = F.relu(y)
            else:
                x = F.relu(self.fc1(state))
            if self.modification:
                y = self.fc2(x)
                if len(y.size())==1:
                    y = y.unsqueeze(0)
                y = self.bn2(y)
                y = y.squeeze(0)
                x = F.relu(y)
            else:
                x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        else:
            x = self.bn0(state)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            return torch.tanh(self.fc3(x))