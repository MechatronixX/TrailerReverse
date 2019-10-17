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
import dqn_model
from dqn_model import DoubleQLearningModel, ExperienceReplay
#from IPython.core.debugger import set_trace



class DDQNhelpers: 
    """Some helper functions for training a DDQN network. Use as a static class  """
    
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

    def calc_q_and_take_action(ddqn, state, eps, actions):
        '''
        Calculate Q-values for current state, and take an action according to an epsilon-greedy policy.
        Inputs:
            ddqn   - DDQN model. An object holding the online / offline Q-networks, and some related methods.
            state  - Current state. Numpy array, shape (1, num_states).
            eps    - Exploration parameter.
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
        
        actionProbs = DDQNhelpers.eps_greedy_policy(q_np, eps)
        
        #Hardcoded action set, unclear where we find the available actions of the ddqn
        
        
        curr_action = np.random.choice(  actions       ,    p=actionProbs) 
        
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
    
        # FYI:
        # ddqn.online_model & ddqn.offline_model are Pytorch modules for online / offline Q-networks, which take the state 
        # as input, and output the Q-values for all actions.
        # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).
    
        # YOUR CODE HERE
        
        #TODO: Something should have no_grad() here
        q_online_next = ddqn.online_model(next_state)
        q_offline_next = ddqn.offline_model(next_state)
        q_online_curr = ddqn.online_model(curr_state)
        
        q_target = calculate_q_targets(q_online_next, q_offline_next, reward, nonterminal, gamma=gamma)
        #q_target.
        loss = ddqn.calc_loss(q_online_curr, q_target.detach(), curr_action)
    
        return loss
    
    #episode_log = namedtuple("episode_log", ["pos_x", "pos_y", "steering_angle"])    
    
    #Static object for now
    #train_log = []; 


    def train_loop_ddqn( env, ddqn, replay_buffer, num_episodes, enable_visualization=False, batch_size=64, gamma=.94):        
        
        Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
       
        eps = 1.
        eps_end = .1 
        eps_decay = .001
        tau = 1000
        cnt_updates = 0
        R_buffer = []
        R_avg = []
        for i in range(num_episodes):
            #set_trace()
            state = env.reset() # Initial state
            
            state = state[None,:] # Add singleton dimension, to represent as batch of size 1.
            finish_episode = False # Initialize
            ep_reward = 0 # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
            q_buffer = []
            posX_log= []
            posY_log = []
            action_log = [] 
            
            steps = 0
            
            maxSteps = 700
            
            #TODO: Define a reasonable break condition, should rather be distance traveled in relation to the initial position. 
            while not finish_episode and steps < maxSteps:
                
                if enable_visualization:
                    env.render() # comment this line out if you don't want to / cannot render the environment on your system
                steps += 1
    
                # Take one step in environment. No need to compute gradients,
                # we will just store transition to replay buffer, and later sample a whole batch
                # from the replay buffer to actually take a gradient step.
                debugac = 0
                q_online_curr, curr_action = DDQNhelpers.calc_q_and_take_action(ddqn, state, eps, debugac)
                q_buffer.append(q_online_curr)
                
                #Debug: Later on actions will be velocity, but for not it is just steering. 
                v = -1
                action = [v, curr_action]
                #new_state, reward, finish_episode, _ = env.step(action) # take one step in the evironment
                
                new_state, reward, finish_episode = env.step(action) # take one step in the evironment
                
                 #Log some. 
                posX_log.append(new_state[0])
                posY_log.append(new_state[1])
                
                new_state = new_state[None,:]
                
               
                action_log.append(curr_action)
                
                # Assess whether terminal state was reached.
                # The episode may end due to having reached 200 steps, but we should not regard this as reaching the terminal state, and hence not disregard Q(s',a) from the Q target.
                # https://arxiv.org/abs/1712.00378
                nonterminal_to_buffer = not finish_episode or steps == maxSteps
                
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
    
                    cnt_updates += 1
                    if cnt_updates % tau == 0:
                        ddqn.update_target_network()
            
            ###########################
            ## Episode finished. 
            
            
            train_log.append( episode_log(pos_x = posX_log, pos_y = posY_log, steering_angle = action_log ) )
              
            eps = max(eps - eps_decay, eps_end) # decrease epsilon        
            R_buffer.append(ep_reward)
            
            #Commented out for debug. 
            # Running average of episodic rewards (total reward, disregarding discount factor)
            R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i-1])  if i > 0 else  R_avg.append(R_buffer[i])
    
            if(i%1 == 0):
                print('Episode: {:d}, Total Reward (running avg): {:4.0f}'.format( i, R_avg[-1]))
            #print('Episode: {:d}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Avg Q: {:.4g}'.format(i, ep_reward, R_avg[-1], eps, np.mean(np.array(q_buffer))))
            
            # If running average > 195 (close to 200), the task is considered solved
            if R_avg[-1] > 195:
                return R_buffer, R_avg
        return R_buffer, R_avg
        
    