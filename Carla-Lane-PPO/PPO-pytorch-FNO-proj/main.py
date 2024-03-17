import sys
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
from PIL import Image


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
   
    
def convert(rgb):
    rgb = rgb.reshape(48, 48, 3)
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def random_resized_crop(img):
    img = np.asarray((img + 1) * 255, dtype=np.uint8)
    im0 = Image.fromarray(img[0])
    im1 = Image.fromarray(img[1])
    im2 = Image.fromarray(img[2])
    im3 = Image.fromarray(img[3])
    im0 = im0.resize((60, 60), resample=Image.BILINEAR)
    im1 = im1.resize((60, 60), resample=Image.BILINEAR)
    im2 = im2.resize((60, 60), resample=Image.BILINEAR)
    im3 = im3.resize((60, 60), resample=Image.BILINEAR)
    
    img = np.array((np.array(im0), np.array(im1), np.array(im2), np.array(im3)))
    xr = int(np.random.rand() * 12)
    yr = int(np.random.rand() * 12)
    img = img[:, xr:xr+48, yr:yr+48] 
    
    return img
    

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

name = f'{env_name}_PPO_FNO_proj'
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
hist = -1000
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
            # s = random_resized_crop(s)
            memory.append([s, a, r, mask])

            score += r
            s = s_
            if done:
                break
        with open('log_carla.txt', 'a') as outfile:
            outfile.write('\t' + str(episodes) + '\t' + str(score) + '\n')
        scores.append(score)
        
    score_avg = np.mean(scores)
    if score_avg > hist:
        hist = score_avg
        torch.save(ppo.actor_net.state_dict(), 'models/actor.pth')
        torch.save(ppo.critic_net.state_dict(), 'models/critic.pth')
        torch.save(ppo.actor_optim.state_dict(), 'models/actor_optim.pth')
        torch.save(ppo.critic_optim.state_dict(), 'models/critic_optim.pth')
        
        print('Model saved')
        
    print('{} episode score is {:.2f}'.format(episodes, score_avg))
    wandb.log({"Episode Score": score_avg}, step=iter)

    ppo.train(memory)
