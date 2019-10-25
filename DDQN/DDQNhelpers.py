# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:08:56 2019

@author: root
"""

# Import dependencies
import torch
import numpy as np
from numpy import array
#import gym
from collections import namedtuple
import DDQN.dqn_model
from DDQN.dqn_model import DoubleQLearningModel, ExperienceReplay
#from IPython.core.debugger import set_trace

def calculate_q_targets(q1_batch, q2_batch, r_batch, nonterminal_batch, gamma=.99):
    '''
    Calculates the Q target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network. FloatTensor, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network. FloatTensor, shape (N, num actions)
    : param r_batch: Batch of rewards. FloatTensor, shape (N,)
    : param nonterminal_batch: Batch of booleans, with False elements if state s' is terminal and True otherwise. BoolTensor, shape (N,)
    : param gamma: Discount factor, float.
    : return: Q target. FloatTensor, shape (N,)
    '''    
    action = np.argmax(q1_batch.detach().numpy(),  1   )
    
    
     #TODO: It seems that the colon operator is different from the matlab(?), so we need to select elements like this instead.
    N = q2_batch.size(0)
    discountTerm = gamma*q2_batch[ range(N) ,action]
    
    #Add discounted value for all states that aren't terminal
    Y =  r_batch 
    Y[nonterminal_batch] =  Y[nonterminal_batch] + discountTerm[nonterminal_batch]
    
    
    return torch.Tensor(Y)

def eps_greedy_policy(q_values, eps):
    '''
    Creates an epsilon-greedy policy
    :param q_values: set of Q-values of shape (num actions,)
    :param eps: probability of taking a uniform random action 
    :return: policy of shape (num actions,)
    '''
    # YOUR CODE HERE
    a_star = np.argmax(q_values)
    
    m = len(q_values)
    policy = np.ones(m )*eps/m 
    
    #Add extra probability that we choose the optimal action
    policy[a_star] = policy[a_star] +1-eps  
    
    return policy

def calc_q_and_take_action(ddqn, state, eps):
    '''
    Calculate Q-values for current state, and take an action according to an epsilon-greedy policy.
    Inputs:
        ddqn   - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        state  - Current state. Numpy array, shape (1, num_states).
        eps    - Exploration parameter.
        Nactions - number of actions available. 
    Returns:
        q_online_curr   - Q(s,a) for current state s. Numpy array, shape (1, num_actions) or  (num_actions,).
        curr_action     - Selected action (0 or 1, i.e. left or right), sampled from epsilon-greedy policy. Integer.
    '''
    # FYI:
    # ddqn.online_model & ddqn.offline_model 
    # are Pytorch modules for online / offline Q-networks, which take the state as input, 
    # and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).
    
    
    # YOUR CODE HERE
    
    #Seems we need to cast the incoming state to a tensor for later function uses. Note that tensor constructor 
    #performs a deep copy

    #state_tensor = torch.tensor(state, requires_grad = False, ).to(device)
    #By detach() the tensor should never need a gradient and is detached from the computational graph it seems
    state_tensor = torch.FloatTensor(state).detach()
    #state_tensor = state_tensor.detach()
    
    q_online_curr = ddqn.online_model( state_tensor )
    
    #set_trace()
    
    #Cast to numpy and remove axtra dimension. 
    q_np = q_online_curr.detach().numpy().squeeze()
    
  
    actionProbs = eps_greedy_policy(q_np, eps)
    
    #The actions are indexes 0,...,nactions-1, and are exactly the as many as the q-values. 
    Nactions = len(q_np)
    actions = range(Nactions)
    
    curr_action = np.random.choice(  Nactions       ,    p=actionProbs) 
    
    return q_online_curr, curr_action

def sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma):
    '''
    Sample mini-batch from replay buffer, and compute the mini-batch loss
    Inputs:
        ddqn          - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        replay_buffer - Replay buffer object (from which smaples will be drawn)
        batch_size    - Batch size
        gamma         - Discount factor
    Returns:
        Mini-batch loss, on which .backward() will be called to compute gradient.
    '''
    # Sample a minibatch of transitions from replay buffer
    curr_state, curr_action, reward, next_state, nonterminal = replay_buffer.sample_minibatch(batch_size)

    #set_trace()
    # FYI:
    # ddqn.online_model & ddqn.offline_model are Pytorch modules for online / offline Q-networks, which take the state 
    # as input, and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).

    # YOUR CODE HERE
    
    #TODO: Something should have no_grad() here
    # with torch.no_grad():
    q_online_next = ddqn.online_model(next_state)
    q_offline_next = ddqn.offline_model(next_state)
    q_online_curr = ddqn.online_model(curr_state)
    
    q_target = calculate_q_targets(q_online_next, q_offline_next, reward, nonterminal, gamma)
    #q_target.
    loss = ddqn.calc_loss(q_online_curr, q_target.detach(), curr_action)

    return loss
    


