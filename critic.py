import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import hidden_init

class Critic(nn.Module):
    def __init__(self, state_size, action_size, modification, seed, fcs1_units=400, fc2_units=300, environment='pendulum'):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.modification = modification
        self.environment = environment
        self.state_size = state_size
        self.action_size = action_size
        if environment=='pendulum':
            self.fcs1 = nn.Linear(state_size, fcs1_units)
            if self.modification:
                self.bn1 = nn.BatchNorm1d(fcs1_units)
            self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
            self.fc3 = nn.Linear(fc2_units, 1)
            self.reset_parameters()
        else:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.fcs1 = nn.Linear(state_size, fcs1_units)
            self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
            self.fc3 = nn.Linear(fc2_units, 1)
            self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        if self.environment == 'pendulum':
            if self.modification:
                y = self.fcs1(state)
                xs = F.relu(self.bn1(y))
            else:
                xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        else:
            state = self.bn0(state)
            xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)