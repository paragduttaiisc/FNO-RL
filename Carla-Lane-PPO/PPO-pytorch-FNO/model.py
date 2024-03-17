import torch
import torch.nn as nn
from fno import FNO2d as FNO
from parameters import *
    
class Actor(nn.Module):
    def __init__(self, N_S, N_A):
        super(Actor,self).__init__()
        # input: 4*48*48
        self.conv = FNO(5, 5, 128) # outputs 512
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.sigma = nn.Linear(64, N_A)
        self.mu = nn.Linear(64, N_A)
        self.distribution = torch.distributions.Normal

    def forward(self, s):
        x = self.conv(s)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # mu = torch.tanh(self.mu(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def choose_action(self, s):
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().numpy()

# Critic
class Critic(nn.Module):
    def __init__(self, N_S):
        super(Critic, self).__init__()
        self.conv = FNO(5, 5, 128) # outputs 512
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, s):
        x = self.conv(s)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
