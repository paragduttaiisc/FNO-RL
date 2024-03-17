import torch
import torch.nn as nn
from parameters import *

# Actor
class Actor(nn.Module):
    def __init__(self, N_S, N_A):
        super(Actor,self).__init__()
        # input: 4*48*48
        self.conv1 = nn.Conv2d(N_S[0], 32, kernel_size=5, stride=2)
        # 32*22*22
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # 64*10*10
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        # 64*4*4
        self.fc1 = nn.Linear(64*4*4, 256)
        # 256
        self.fc2 = nn.Linear(256,64)
        self.sigma = nn.Linear(64, N_A)
        self.mu = nn.Linear(64, N_A)
        self.distribution = torch.distributions.Normal

    def forward(self, s):
        x = torch.relu(self.conv1(s))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        mu = self.mu(x)
        log_sigma = self.sigma(x)
        #log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self, s):
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().numpy()

# Critic
class Critic(nn.Module):
    def __init__(self, N_S):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(N_S[0], 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, s):
        x = torch.relu(self.conv1(s))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x