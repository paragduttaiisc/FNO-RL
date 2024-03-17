import gym
import d4rl
import torch
import numpy as np
from parameters import *
from PPO import Ppo
from tqdm import tqdm
from collections import deque
from gym.wrappers import FrameStack
import matplotlib.pyplot as plt
import wandb


def convert(rgb):
    rgb = rgb.reshape(48, 48, 3)
    rgb = rgb.transpose(2, 0, 1)
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return rgb


class RGB2Grey(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return convert(obs)


class Normalize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs = np.array(obs)
        obs = obs.reshape(-1, 48, 48)
        return np.array(obs) * 2 - 1


env_name = 'carla-lane-v0'
env = gym.make(env_name)
env = RGB2Grey(env)
env = FrameStack(env, 4)
env = Normalize(env)

# obs = env.reset()
# print(obs.shape)
# import sys; sys.exit()
# random seeds
np.random.seed(2024)
torch.manual_seed(2024)
name = f'{env_name}_PPO_CNN'
wandb.init(project='CARLA_PPO', entity='mbrl-nfo', name=name)

wandb.config.lr_actor = lr_actor
wandb.config.lr_critic = lr_critic
wandb.config.Iter = Iter
wandb.config.MAX_STEP = MAX_STEP
wandb.config.gamma = gamma
wandb.config.lambd = lambd
wandb.config.batch_size = batch_size
wandb.config.epsilon = epsilon
wandb.config.l2_rate = l2_rate
wandb.config.beta = beta
# class Nomalize:


#     def __init__(self, N_S):
#         self.mean = np.zeros((N_S,))
#         self.std = np.zeros((N_S,))
#         self.stdd = np.zeros((N_S,))
#         self.n = 0
#
#     def __call__(self, x):
#         x = np.asarray(x)
#         self.n += 1
#         if self.n == 1:
#             self.mean = x
#         else:
#             old_mean = self.mean.copy()
#             self.mean = old_mean + (x - old_mean) / self.n
#             self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
#         if self.n > 1:
#             self.std = np.sqrt(self.stdd / (self.n - 1))
#         else:
#             self.std = self.mean
#         x = x - self.mean
#         x = x / (self.std + 1e-8)
#         x = np.clip(x, -5, +5)
#         return x


ppo = Ppo((12, 48, 48), 2) # 4*3, 48, 48
episodes = 0
eva_episodes = 0
for iter in range(Iter):
    memory = deque()
    scores = []
    steps = 0
    while steps < 2048:  # Horizon
        episodes += 1
        s = env.reset()
        score = 0
        for _ in range(MAX_STEP):
            steps += 1
            # on policy actions
            a = ppo.actor_net.choose_action(torch.from_numpy(
                np.array(s).astype(np.float32)).unsqueeze(0))[0]
            s_, r, done, info = env.step(a)

            mask = (1 - done) * 1
            memory.append([s, a, r, mask])

            score += r
            s = s_
            if done:
                break
        with open('log_carla.txt', 'a') as outfile:
            outfile.write('\t' + str(episodes) + '\t' + str(score) + '\n')
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(episodes, score_avg))
    wandb.log({"Episode Score": score_avg}, step=iter)

    ppo.train(memory)
