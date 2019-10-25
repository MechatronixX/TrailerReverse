# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:06:25 2019

@author: root
"""

import gym
from PPO.PPO import PPO, Memory
from PIL import Image
import torch
from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv

##Simulate pretrained PPO on an environment.  
env = CarTrailerParkingRevEnv()

def runPPO():
    ############## Hyperparameters ##############
    #env_name = "LunarLander-v2"
    # creating environment
    #env = gym.make(env_name)
    #state_dim = env.observation_space.shape[0]
    #action_dim = 4
    
    #env_name = "LunarLander-v2"
    env_name = "TrailerReversingDiscrete"
    # creating environment
    #env = gym.make(env_name)
    
    #state_dim = env.observation_space.shape[0]
    state_dim = 7
    action_dim = 4
    
    render = True
    max_timesteps = 619
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 3
    max_timesteps = 700
    render = True
    save_gif = False

    filename = "PPO_{}.pth".format(env_name)
    directory = "./preTrained/"
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            #action = 1 #Debug by forcing an action 
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
    
if __name__ == '__main__':
    try:
        runPPO()
    #By doing this the animation shouldnt cause a crash. 
    except KeyboardInterrupt:
        print('Interrupted')
        env.close()
    
    
    