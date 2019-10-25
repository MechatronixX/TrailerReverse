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

#Load a pretrained model from disc and continue training it 
usePretrained = True

#Actions are full turn left, straight, full turn right
#actions = (-1,0,1)

env = CarTrailerParkingRevEnv()

device = torch.device("cpu")

# Initializations

num_actions = env.action_space.n
num_states = env.observation_space.shape[ 0]

print('Training an environment with number of actions: ', num_actions, 'Number of states:', num_states)

#Training hyperparameters. 
num_episodes = 1000
batch_size = 128
gamma = .94
learning_rate = 1e-3

# Object holding our online / offline Q-Networks
ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate)

if usePretrained: 
    #Path where to put network parameters.  
    directory = "preTrained"
    filename = './DDQN_{}.pth'.format(env.name)
    
    print('Loading a pretrained DDQN model. Results')
    
    #Load a pretrained model. This should make sense for this off policy approach??
    loadResult1 = ddqn.online_model.load_state_dict(torch.load(directory + filename))
    loadResult2 = ddqn.offline_model.load_state_dict(torch.load(directory+filename))
    
    print(loadResult1,loadResult2)
    
#We should train for a time long enough to converge to our target. 
initState = env.state
distToTarget = np.sqrt( initState[0]**2 + initState[1]**2  )
timeToTarget = env.max_speed/distToTarget*200   #Nominal birdsflight time at max speed times margin
maxSteps = np.round(timeToTarget/env.dt) 

print('Will train each episode for (in real time)', timeToTarget, ' seconds implying ', maxSteps, ' discrete steps') 

# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored 
# for training
replay_buffer = ExperienceReplay(device, num_states)

# Train
trainingLog = train_loop_ddqn( env, ddqn, replay_buffer, num_episodes, enable_visualization, batch_size, gamma, maxSteps)



