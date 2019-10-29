# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:33:53 2019

@author: root
"""

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