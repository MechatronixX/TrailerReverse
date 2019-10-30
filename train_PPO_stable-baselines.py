#!/usr/bin/env python
# coding: utf-8

################################################################################
# This is the main method for training the PPO algorithm from stable-baselines #
# Methods: callback()                                                          #
#          train()                                                             #
#                                                                              #
# callback()                                                                   #
# This function has potential that is not used yet. Feel free to improve it    #
# Inputs: _                                                                    #
# Outputs: _                                                                   #
#                                                                              #
# train()                                                                      #
# The train function initializes the training of the open ai gym environment   #
# Inputs: _                                                                    #
# Outputs: _                                                                   #
################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from gymEnvironments.gym_reverse_variable_trailer_number import Reverse_variable_trailer_number_environment

def callback(_locals, _globals):    
    return True 

def train():
    # multiprocess environment
    n_cpu = 4
    env = SubprocVecEnv([lambda: Reverse_variable_trailer_number_environment() for i in range(n_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1, learning_rate=1e-5)
    model.learn(total_timesteps=np.int(1e9), log_interval=10, callback = callback)
    model.save("ppo2_trailer")
    
if __name__ == "__main__":
    train()