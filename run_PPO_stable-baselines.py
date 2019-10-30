#!/usr/bin/env python
# coding: utf-8

##########################################################################################
# run()                                                                                  # 
# This is the main method of the functionality to visualize the outcomes of the training #
# Inputs: _                                                                              #
# Outputs: _                                                                             #
##########################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

from stable_baselines import PPO2

from gymEnvironments.gym_reverse_variable_trailer_number import Reverse_variable_trailer_number_environment

def run():
    model = PPO2.load("ppo2_trailer")
    env = Reverse_variable_trailer_number_environment()
    obs = env.reset()
    done = False 
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, _ = env.step(action)
        env.render()
        
if __name__ == "__main__":
    run()