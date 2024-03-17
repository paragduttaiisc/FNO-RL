import torch
import torch.nn as nn
from fno import FNO2d as FNO
from parameters import *
    
class Actor(nn.Module):
    def __init__(self, N_S, N_A):
        super(Actor,self).__init__()
        # input: 4*48*48
        self.conv = FNO(5, 5, 128) # outputs 512
        self.fc = nn.Linear(512, 100)
        self.trunk = nn.Sequential(
            torch.nn.Linear(100, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )
        self.sigma = nn.Linear(1024, N_A)
        self.mu = nn.Linear(1024, N_A)
        self.distribution = torch.distributions.Normal
        self.log_std_min = -10
        self.log_std_max = 2
        
    def forward(self, s):
        x = self.conv(s)
        x = torch.tanh(self.fc(x))
        x = self.trunk(x)
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        
        # log_sigma = torch.tanh(log_sigma)
        # log_sigma = self.log_std_min + 0.5 * (
        #     self.log_std_max - self.log_std_min
        # ) * (log_sigma + 1)
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
        self.conv = FNO(5, 5, 128)
        self.fc = nn.Linear(512, 100)
        self.trunk = nn.Sequential(
            torch.nn.Linear(100, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        )

    def forward(self, s):
        x = self.conv(s)
        x = torch.tanh(self.fc(x))
        x = self.trunk(x)
        return x
