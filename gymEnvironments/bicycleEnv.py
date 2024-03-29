# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:23:53 2019

@author: root
"""
import numpy as np
from numpy import array
import copy 
from collections import namedtuple

#TODO: Create a proper gym environment when we feel ready. 
#class CustomEnv(gym.Env):
class BicycleEnv: 
    """A simple bicyle model"""
    #metadata = {'render.modes': ['human']}

    
    
    def __init__(self, L=2, Ts=0.05, initState= (5,0,0) ):
        """ L is length front wheel center to reear wheel center """
          
        self.L = L
        self.Ts = Ts
          
        #Position and orientation 
        self.Px = initState[0]
        self.Py = initState[1]
        self.heading = initState[2]
        
        self.initState = copy.deepcopy(initState)
        
        self.actionTuple = namedtuple("Action", ["vel", "steeringRad"])
        
        V0 = 1
        
        self.action_map = {0:  self.actionTuple(vel=V0, steeringRad = np.deg2rad(-45)  ), 
                           1:  self.actionTuple(vel=V0, steeringRad = np.deg2rad(0)    ),
                           2:  self.actionTuple(vel=V0, steeringRad = np.deg2rad(45)   ),
                           3:  self.actionTuple(vel=0, steeringRad = 0                    )}                                                           
        
        #self.actions = (0,1,2)
        #self.state = copy.deepcopy(initstate)
      
     
    
    
    def __calculateReward__(self):
        
        reward = -( self.Px**2 + self.Py**2 )
        #Simple reward: Try to get the agent to figure out how to reach the x axis. 
        #reward = -self.Px**2
        return reward
    
    def step(self, a):
      """Action is a (tuple velocity, steering angle) """
       
      action = self.action_map[a]
      
      self.__eulerForwardStep__(action.vel, action.steeringRad)
      
      reward = self.__calculateReward__()
      
      distToTarget = np.sqrt( self.Px**2 + self.Py**2 )
      #distToTarget = np.sqrt( self.Px**2 )
      episode_finished = distToTarget < 0.1
      
      return self.__getstate__(), reward, episode_finished
      
      
      #print(self.Py)
      
     
      return self.__getstate__()
  
    def __eulerForwardStep__(self,V, steeringAngle):
      qdot = np.tan(steeringAngle)/self.L*V
      
      Ts = self.Ts 
      
      self.heading = self.heading + qdot*Ts 
      
      vx = -np.sin(self.heading)*V
      vy = np.cos(self.heading)*V
      
      self.Px = self.Px + vx*Ts 
      self.Py = self.Py + vy*Ts

      
      
    # Execute one time step within the environment
    def reset(self):
        # Reset the state of the environment to an initial state
        
        #TODO : Use a named tuple for all these??
        self.Px         = copy.deepcopy(self.initState[0])
        self.Py         = copy.deepcopy(self.initState[1])
        self.heading    = copy.deepcopy(self.initState[2])
        
        return self.__getstate__()
    
    def __getstate__(self): 
        #Do not get external callers the possibility to fiddle with the internal state. 
        return copy.deepcopy( np.array( [self.Px, self.Py, self.heading] ))
        
       #Try to instantiate  
##L = 2 #Length rear axis to front axis
#Ts = 0.01 #Sample interval 

#Position x, y, and heading
#initState = (5,1, 0)
#bicycle = BicycleEnv(L,Ts, initState)
   
    #def render(self, mode='human', close=False):
        # Render the environment to the screen
 