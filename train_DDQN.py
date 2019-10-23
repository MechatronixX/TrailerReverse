# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:12:56 2019

@author: root
"""

# Import dependencies
import torch
import numpy as np
from numpy import array
import gym
from collections import namedtuple
import DDQN
from DDQN import DDQNhelpers
from DDQN.DDQN_trainLoop import train_loop_ddqn
from DDQN.dqn_model import DoubleQLearningModel, ExperienceReplay
import DDQN.dqn_model
from DDQN.DDQNhelpers import *

from IPython.core.debugger import set_trace
from gymEnvironments.bicycleEnv import *
from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv

# Enable visualization? Does not work in all environments.
enable_visualization = False

#Actions are full turn left, straight, full turn right
#actions = (-1,0,1)

env = CarTrailerParkingRevEnv()

device = torch.device("cpu")

# Initializations

#num_actions = len(truck.action_map.keys()) #TODO: Hardcoded now, do something more fancy later.
#num_states = len(truck.initState)

num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

print('Number of actions', num_actions, 'Number of states', num_states)
#Training hyperparameters. 
num_episodes = 700
batch_size = 128
gamma = .94
learning_rate = 1e-3

# Object holding our online / offline Q-Networks
ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate)


# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored 
# for training
replay_buffer = ExperienceReplay(device, num_states)

# Train
trainingLog = train_loop_ddqn( env, ddqn, replay_buffer, num_episodes, enable_visualization, batch_size, gamma)



