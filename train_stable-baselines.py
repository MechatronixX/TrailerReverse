# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:07:54 2019

@author: root
"""


#Try to the train a system using stable baselines. Example from 
#https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv


def callback(_locals, _globals):
    #TODO: This callback could implement a reset for timeout? So an episode doesnt go on for too long 
    # when diverging?? 
    
    return True 

def train():
    #TODO: Can our system be trained here right off? 
    # multiprocess environment
    n_cpu = 8
    #env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
    env = SubprocVecEnv([lambda: CarTrailerParkingRevEnv()  for i in range(n_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=500000, log_interval=10, callback = callback)
    model.save("ppo2_trailer")

    #del model # remove to demonstrate saving and loading

    #model = PPO2.load("ppo2_trailer")

    # Enjoy trained agent
    #obs = env.reset()
    #while True:
    #    action, _states = model.predict(obs)
    #    obs, rewards, dones, info = env.step(action)
    #    env.render()
        
#It is important that the file is called using the contruct below
#https://github.com/hill-a/stable-baselines/issues/155        
if __name__ == "__main__":
    train()