# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:23:05 2019

@author: root

Run a pre traind DDQN model for the trailer reverse problem. 
"""

#Run a pre trained DDQN model for this problem. 

#import DDQN
from DDQN import DDQNhelpers
from DDQN.DDQN_trainLoop import train_loop_ddqn
from DDQN.dqn_model import DoubleQLearningModel, ExperienceReplay
import DDQN.dqn_model
from DDQN.DDQNhelpers import *

from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv

import time
num_episodes = 10
#env = gym.make("CartPole-v0")

env = CarTrailerParkingRevEnv()

device = torch.device("cpu")

num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
learning_rate = 1e-3
ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate)

ddqn.online_model.load_state_dict(torch.load("preTrained/DDQN_TrailerReversingDiscrete.pth")  )

maxSteps = 300

#Simulation pause. Scale up or down to make simulation slower or faster. 
Ts = env.dt
Ts_anim = Ts/10

for i in range(num_episodes):
        state = env.reset() #reset to initial state
        state = state[None,:]
        terminal = False # reset terminal flag
        
        for i in range(maxSteps):
            env.render()
            time.sleep(Ts_anim)
            with torch.no_grad():
                q_values = ddqn.online_model(torch.tensor(state, dtype=torch.float, device=device)).cpu().numpy()
            policy = eps_greedy_policy(q_values.squeeze(), .1) # greedy policy
            action = np.random.choice(num_actions, p=policy)
            state, reward, terminal, _ = env.step(action) # take one step in the evironment
            state = state[None,:]
            if terminal: 
                break
# close window
env.close();