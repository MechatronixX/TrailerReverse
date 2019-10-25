# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:30:43 2019

@author: root
"""

import torch
import numpy as np
from numpy import array
from collections import namedtuple
from DDQN.DDQNhelpers import *


def train_loop_ddqn( env, ddqn, replay_buffer, num_episodes, enable_visualization=False, batch_size=64, gamma=.94, maxSteps = None):        
    """ A loop that can be used to train a DDQN network. Will usually be somewhat application dependant,
        but this is a baseline that could/should work for the trailer-vehicle environment. """
    
    #File log output parameters
    env_name = "TrailerReversingDiscrete" #TODO: Should be polled from the environment. 
    directory = "./preTrained/"
    filename = './DDQN_{}.pth'.format(env_name)
    
    #Steps per episode . Revert to default if non argument explicitly provided. 
    if(maxSteps == None):
        maxSteps = 500
    
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    #Things we log per each time step each episode. 
    perStepLogTuple = namedtuple("perStepLog", ["Px", "Py","angleRad", "action"])
    
    #Nesteld tuples 
    perEpisodeLogTuple = namedtuple("perEpLog", ["perStepLog", "eps", "R", "Ravg"])
    
    trainingLog = perEpisodeLogTuple(perStepLog = [], eps = [], R=[], Ravg=[] )
    
    #Initial and final probability of taking a random action
    eps = 0.99
    eps_end = 0.1 
    
    #Epsilon decays this much each apisode
    #eps_decay = .01
    eps_decay = 0.001
    tau = 1000
    cnt_updates = 0
    R_buffer = []
    R_avg = []
    
    print('Starting to train.')
    
    for i in range(num_episodes):
        state = env.reset() # Initial state
        state = state[None,:] # Add singleton dimension, to represent as batch of size 1.
        finish_episode = False # Initialize
        ep_reward = 0 # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
        q_buffer = []
        steps = 0
        
        
        perStepLog = perStepLogTuple(Px =[], Py =[], action =[], angleRad = [])
        
        while not finish_episode and steps < maxSteps:
           
            if enable_visualization:
                env.render() # comment this line out if you don't want to / cannot render the environment on your system
            steps += 1

            # Take one step in environment. No need to compute gradients,
            # we will just store transition to replay buffer, and later sample a whole batch
            # from the replay buffer to actually take a gradient step.
            q_online_curr, curr_action = calc_q_and_take_action(ddqn, state, eps)
            q_buffer.append(q_online_curr)
            
            #Velocity is constant for now
            new_state, reward, finish_episode,_  = env.step(curr_action) # take one step in the evironment
            
            #This is hardcoded for the current state definition, so the log will not make sense for other gym environments. 
            perStepLog.Px.append(new_state[0] )
            perStepLog.Py.append(new_state[1] )
            perStepLog.angleRad.append(new_state[2] )
            perStepLog.action.append(curr_action)
            
            #set_trace()
            
            new_state = new_state[None,:]
            
            
            # Assess whether terminal state was reached.
            # The episode may end due to having reached 200 steps, but we should not regard this as reaching the terminal state, and hence not disregard Q(s',a) from the Q target.
            # https://arxiv.org/abs/1712.00378
            
            #Timeout should not be a termination of episode, that will create partially observable markov
            #which is harder o train. 
            #nonterminal_to_buffer = not finish_episode or steps == maxSteps
            nonterminal_to_buffer = not finish_episode 
            
            # Store experienced transition to replay buffer
            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

            state = new_state
            ep_reward += reward
            
            # If replay buffer contains more than 1000 samples, perform one training step
            if replay_buffer.buffer_length > 1000:
                
                loss = sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma)
                ddqn.optimizer.zero_grad()
                loss.backward()
                ddqn.optimizer.step()
                
                #Transfer online network parameters to offline network. 
                cnt_updates += 1
                if cnt_updates % tau == 0:
                    ddqn.update_target_network()
        #########################
        ## End of episode
        
        if i % 100 ==0 and i >0: 
            print("Saving DDQN parameters to disk.")
            torch.save(ddqn.online_model.state_dict(), directory+filename)
        
        eps = max(eps - eps_decay, eps_end) # decrease epsilon        
        
        R_buffer.append(ep_reward)
         
        # Running average of episodic rewards (total reward, disregarding discount factor)
        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i-1])  if i > 0 else  R_avg.append(R_buffer[i])
        
        trainingLog.perStepLog.append(perStepLog)
        trainingLog.eps.append(eps)
        trainingLog.Ravg.append(R_avg[-1])
        trainingLog.R.append(ep_reward )

        if(i%10 == 0):
            print('Episode: {:d}, Total Reward (running avg): {:4.0f}, Epsilon: {:.3f}'.format( i, R_avg[-1], eps))
        #print('Episode: {:d}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Avg Q: {:.4g}'.format(i, ep_reward, R_avg[-1], eps, np.mean(np.array(q_buffer))))
        
        # If running average > 195 (close to 200), the task is considered solved
        #if R_avg[-1] > 195:
        #    return trainingLog
    #########################
    ### Enable training
    if enable_visualization: 
        env.close()
    print("Training finished.")
    return trainingLog