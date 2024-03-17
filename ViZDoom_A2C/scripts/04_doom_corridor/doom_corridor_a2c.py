import numpy as np
import random                # Handling random number generation
import time                  # Handling time calculation
import cv2
import wandb

import torch
from vizdoom import *        # Doom Environment
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import namedtuple, deque
import math

import sys
sys.path.append('../../')
from algos.agents import A2CAgent
from algos.models import ActorCnn, CriticCnn, FNO2d, Actor, Critic
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (0, -60, -40, 60), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

def create_environment():
    game = DoomGame()
    game.load_config("doom_files/deadly_corridor.cfg")
    game.set_doom_scenario_path("doom_files/deadly_corridor.wad")
    possible_actions  = np.identity(7, dtype=int).tolist()
    return game, possible_actions

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    game.set_window_visible(False)
    game.init()
    for i_episode in range(start_epoch + 1, n_episodes+1):
        game.new_episode()
        state = stack_frames(None, game.get_state().screen_buffer.transpose(1, 2, 0), True) 
        score = 0
        while True:
            action, log_prob, entropy = agent.act(state)
            reward = game.make_action(possible_actions[action])
            done = game.is_episode_finished()
            score += reward
            if done:
                break
            else:
                next_state = stack_frames(state, game.get_state().screen_buffer.transpose(1, 2, 0), False)
                agent.step(state, log_prob, entropy, reward, done, next_state)
                state = next_state
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        wandb.log({"Episode Score": score}, step=i_episode)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        
    game.close()
    
    return scores



game, possible_actions = create_environment()
game.set_window_visible(False)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

name = f'Vizdoom_FNO_512_0.1_0.5_50'
wandb.init(project="VIZDOOM", entity="mbrl-nfo", name=name)
config = wandb.config
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)

SEED = 2024
GAMMA = 0.99           # discount factor
ALPHA= 0.0003         # Actor learning rate
BETA = 0.0003          # Critic learning rate
UPDATE_EVERY = 50     # how often to update the network 

wandb.config.SEED = SEED
wandb.config.GAMMA = GAMMA           # discount factor
wandb.config.ALPHA = ALPHA         # Actor learning rate
wandb.config.BETA = BETA          # Critic learning rate
wandb.config.UPDATE_EVERY = UPDATE_EVERY     # how often to update the network

agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)

scores = train(1000)


