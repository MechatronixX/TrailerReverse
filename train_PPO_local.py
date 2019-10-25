# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:29:28 2019

@author: root
"""

import torch
#import torch.nn as nn
#from torch.distributions import Categorical
from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv
#from PPO.PPO_continuous import Memory
from PPO.PPO import PPO, Memory
import os.path

env = CarTrailerParkingRevEnv()

#Train the PPO to control the discrete trailer/truck environment
 
def trainDiscreteTrailerTruck():
    ############## Hyperparameters ##############
    #env_name = "LunarLander-v2"
    env_name = "TrailerReversingDiscrete"
    
    #Loads a pretrainedmodel and then continues training if true 
    usePretrainedModel = False
    
    #Show a visualization during training. 
    enableVisualization = False 
    # creating environment
    #env = gym.make(env_name)
    
    
     #File log output parameters 
    directory = "./preTrained/"
    filename = './PPO_{}.pth'.format(env_name)
    
    env = CarTrailerParkingRevEnv()
    #state_dim = env.observation_space.shape[0]
    state_dim = 7
    action_dim = 4
    #action_dim = env.action_space.shape[0]
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 5000        # max training episodes
    max_timesteps = 619         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    
    #Update policy after this many episodes. Important hyperparameter, we do not want to a few episodes
    #in each training batch.
    update_after_N_episodes = 7  
    
    update_timestep = update_after_N_episodes*max_timesteps      # update policy every n timesteps
    lr = 3e-4                   #This learning rate was used in the PPO paper  0.0002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 12               # Go over each mini batch of data this many epochs. 
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    #Load a pretrained model if one such exists and it was requested. 
    if(usePretrainedModel and os.path.exists(directory+filename) ): 
        print("Using pretrained model")
        loadRes = ppo.policy.load_state_dict( torch.load( directory+filename )   ) 
        print(loadRes)
    
    print("Learning rate: ", lr, "Beta:", betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    print('Started training. ')
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            
            if enableVisualization:
                env.render()
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), directory+filename)
            break
        
        if i_episode % 100 ==0: 
            print("Saving PPO parameters to disk.")
            torch.save(ppo.policy.state_dict(), directory+filename)
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t average reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
    ##################
    ##End of training
    
    print("End of training") 
    if enableVisualization:
        env.close()
            
if __name__ == '__main__':
    try:
        trainDiscreteTrailerTruck()
    #By doing this the animation shouldnt cause a crash. 
    except KeyboardInterrupt:
        print('Interrupted')
        env.close()
    
    