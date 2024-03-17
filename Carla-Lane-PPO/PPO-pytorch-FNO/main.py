import gym
import sys
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
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class RGB2Grey(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return convert(obs)


class Normalize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.array(obs) * 2 - 1


env_name = 'carla-lane-v0'
env = gym.make(env_name)
env = RGB2Grey(env)
env = FrameStack(env, 4)
env = Normalize(env)

# random seeds
seed = int(sys.argv[1])
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
name = f'{env_name}_PPO_FNO'
wandb.init(project='CARLA_PPO', entity='mbrl-nfo', name=name)

wandb.config.seed = seed
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


ppo = Ppo((4, 48, 48), 2)
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
        for _ in (range(MAX_STEP)):
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
