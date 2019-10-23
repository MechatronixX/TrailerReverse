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

# Enable visualization? Does not work in all environments.
enable_visualization = False

#Actions are full turn left, straight, full turn right
#actions = (-1,0,1)


##System initial conditions. 
L = 2 #Length rear axis to front axis
Ts = 0.2 #Sample interval in seconds. 

#Position x, y, and heading
initState = (5,0, 0)

#Yes, the truck is a bicycle these days. 
truck = BicycleEnv(L,Ts, initState)

device = torch.device("cpu")


# Initializations

num_actions = len(truck.action_map.keys()) #TODO: Hardcoded now, do something more fancy later.
num_states = len(truck.initState)

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
trainingLog = train_loop_ddqn( truck, ddqn, replay_buffer, num_episodes, enable_visualization, batch_size, gamma)



