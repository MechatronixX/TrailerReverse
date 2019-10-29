# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:33:53 2019

@author: root
"""

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv

def run():
    model = PPO2.load("ppo2_trailer")
    n_cpu = 8
    env = CarTrailerParkingRevEnv()
    obs = env.reset()
    done = False 
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, _ = env.step(action)
        env.render()
    
#It is important that the file is called using the contruct below
#https://github.com/hill-a/stable-baselines/issues/155        
if __name__ == "__main__":
    run()