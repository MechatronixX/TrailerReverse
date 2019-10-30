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
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

import numpy as np
import matplotlib.pyplot as plt 
import os

from gymEnvironments.gym_reverse_variable_trailer_number import Reverse_variable_trailer_number_environment

from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv

#Train the in environment and monitor training. Based Example from
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

def callback(_locals, _globals): 
    #print(_locals, type(_locals))
    
    #So the callback returns locals() and globals() from the learn() function in
    #https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py
    #About locals/globals here: https://thepythonguru.com/python-builtin-functions/locals/
    #print("Hallu", _locals['timesteps'])
    #_locals['self'].save(log_dir + 'best_model.pkl')
    return True 


def make_env():
    """ Helper for creating the environment. Put environment in monitor wrapper for logging. """ 
    env = CarTrailerParkingRevEnv() 
    #env =  Reverse_variable_trailer_number_environment()
    
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir, allow_early_resets=True)
    
    return env
    

def train():
    # multiprocess environment
   
    n_cpu = 8
    #env = SubprocVecEnv([lambda: Reverse_variable_trailer_number_environment() for i in range(n_cpu)])
    env = SubprocVecEnv([lambda: make_env() for i in range(n_cpu)])
    
    #Choose either the truck or the car environment. 
    env = CarTrailerParkingRevEnv() 
    #env =  Reverse_variable_trailer_number_environment()
    
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    
    #env = SubprocVecEnv( [env])
    
    #Use tensorboard to get real nice log output live in your web browser: 
    # https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html
    #https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
    #model = PPO2(MlpPolicy, env, verbose=1, learning_rate=0.000025, tensorboard_log="logs/")
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="logs/")
    
    
    model.learn(total_timesteps=np.int(1e6), 
                log_interval=10, 
                callback = callback)
    model.save("ppo2_trailer")
    
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    #y = moving_average(y, window=50)
    # Truncate x
    #x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Filtered")
    plt.show()
    
if __name__ == "__main__":
    train()
    plot_results(log_dir) 