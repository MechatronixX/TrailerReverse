# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:23:53 2019

@author: root
"""
import numpy as np
from numpy import array
import copy 

#TODO: Create a proper gym environment when we feel ready. 
#class CustomEnv(gym.Env):
class BicycleEnv: 
    """A simple bicyle model"""
    #metadata = {'render.modes': ['human']}

    def __init__(self, L, Ts, initState ):
        """ L is length front wheel center to reear wheel center """
          
        self.L = L
        self.Ts = Ts
          
        #Position and orientation 
        self.Px = initState[0]
        self.Py = initState[1]
        self.heading = initState[2]
        
        self.initState = copy.deepcopy(initState)
        #self.state = copy.deepcopy(initstate)
      
     
    def step(self, action):
      """Action is a (tuple velocity, steering angle) """
      
      V = action[0]
      steeringAngle = action[1]
      
      qdot = np.tan(steeringAngle)/self.L*V
      
      Ts = self.Ts 
      
      self.heading = self.heading + qdot*Ts 
      
      vx = -np.sin(self.heading)*V
      vy = np.cos(self.heading)*V
      
      self.Px = self.Px + vx*Ts 
      self.Py = self.Py + vy*Ts
      
      #print(self.Py)
      
     
      return self.__getstate__()
      
      
    # Execute one time step within the environment
    def reset(self):
        # Reset the state of the environment to an initial state
        
        self.Px         = copy.deepcopy(self.initState[0])
        self.Py         = copy.deepcopy(self.initState[1])
        self.heading    = copy.deepcopy(self.initState[2])
        
        return self.__getstate__()
    
    def __getstate__(self): 
        #Do not get external callers the possibility to fiddle with the internal state. 
        return copy.deepcopy( (self.Px, self.Py, self.heading) )
        
       #Try to instantiate  
##L = 2 #Length rear axis to front axis
#Ts = 0.01 #Sample interval 

#Position x, y, and heading
#initState = (5,1, 0)
#bicycle = BicycleEnv(L,Ts, initState)
   
    #def render(self, mode='human', close=False):
        # Render the environment to the screen
 