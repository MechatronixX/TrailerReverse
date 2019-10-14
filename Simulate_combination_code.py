#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python

###################################################################################################################
# Simulate_combination()                                                                                          #  
#                                                                                                                 #
# This class initialises the simulation of of a truck with a variable number of trailers                          #                
# Inputs: destination_translation absolute translation of the destination                                         #
#         destination_rotation    absolute rotation of the destination, value in degrees                          #
# Outputs: _                                                                                                      #
# Methods: run(velocity,steering_percentage)                                                                      #
#                                                                                                                 #
# run(velocity,steering_percentage)                                                                               #
# This method is used to simulate the movement of a truck with a variable number of trailers                      #
# Inputs: destination_translation absolute translation of the destination                                         #
#         steering_percentage     steering angle of the truck, value between -1 and 1                             #
# Outputs: truck_translation       absolute translation of the truck                                              #
#          truck_rotation          absolute rotation of the truck, value in degrees                               #
#          first_trailer_rotation  relative rotation of the first trailer to the truck, value in degrees          #
#          second_trailer_rotation relative rotation of the second trailer to the first trailer, value in degrees #
###################################################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__copyright__ = "Copyright 2019, Chalmers University of Technology"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np
from numpy import array

from vector_rotation_code import *
from constant_rotation_code import *
from endpoint_movement_code import *
from angle_two_vectors_code import *

class Simulate_combination():
    def __init__(self,\
                 truck_translation,\
                 truck_rotation,\
                 first_trailer_rotation,\
                 second_trailer_rotation,\
                 destination_translation,\
                 destination_rotation,\
                 number_second_trailers,\
                 step_size):
        
        self.maximal_first_trailer_rotation = 90
        self.maximal_second_trailer_rotation = 179
        self.maximal_both_trailers_rotation = 181
        
        self.wheelbase_truck = 3.5
        self.maximal_steering_angle = 60
        self.hitch_translation_truck = -1.5
        self.hitch_translation_first_trailer_truck = 2.5
        self.hitch_translation_first_trailer_second_trailer = -0.5
        self.hitch_translation_second_trailer = 5.5
        
        self.truck_translation = truck_translation
        self.truck_rotation = truck_rotation
        self.first_trailer_rotation = first_trailer_rotation
        self.second_trailer_rotation = second_trailer_rotation
        self.destination_translation = destination_translation
        self.destination_rotation = destination_rotation
        self.number_second_trailers = number_second_trailers
        self.step_size = step_size

    def run(self,velocity,steering_percentage):
        steering_angle = steering_percentage*self.maximal_steering_angle
        distance = velocity*self.step_size
        
        truck_movement = constant_rotation(distance,self.truck_rotation)
        first_trailer_movement = vector_rotation(truck_movement,self.first_trailer_rotation)
        second_trailer_movement = vector_rotation(first_trailer_movement,self.second_trailer_rotation)
        
        hitch_vector_truck = constant_rotation(self.hitch_translation_truck,                                               self.truck_rotation)
        hitch_vector_first_trailer_truck = constant_rotation(self.hitch_translation_first_trailer_truck,                                                     self.truck_rotation+                                                     self.first_trailer_rotation)
        hitch_vector_first_trailer_second_trailer = constant_rotation(self.hitch_translation_first_trailer_second_trailer,                                                       self.truck_rotation+                                                       self.first_trailer_rotation)
        hitch_vector_second_trailer = constant_rotation(self.hitch_translation_second_trailer,                                                 self.truck_rotation+                                                 self.first_trailer_rotation+                                                 self.second_trailer_rotation)
        
        if norm(truck_movement) != 0:
            
            old_truck_rotation = self.truck_rotation
            
            rotation_truck = 0      
            ## Truck movement
            if steering_percentage != 0:
                lock = self.wheelbase_truck/np.tan(np.deg2rad(steering_angle))
                rotation_truck = np.rad2deg(np.arcsin(distance/lock))/2
            self.truck_rotation += rotation_truck
            
            translation_truck = vector_rotation(truck_movement,rotation_truck)
            self.truck_translation += translation_truck
          
            step_rotation_truck = self.truck_rotation-old_truck_rotation
            old_first_trailer_rotation = self.first_trailer_rotation
            
            if self.number_second_trailers >= 1:
            
                ## first_trailer movement
                truck_hitch_movement = truck_movement                                       +endpoint_movement(hitch_vector_truck,                                                          step_rotation_truck)
                
                first_trailer_movement = vector_rotation(truck_movement,self.first_trailer_rotation)
                
                rotation_first_trailer = np.sign(self.first_trailer_rotation)                                         *angle_two_vectors(hitch_vector_first_trailer_truck                                         -first_trailer_movement                                         +truck_hitch_movement,                                         np.sign(distance)*truck_movement)
                self.first_trailer_rotation = -step_rotation_truck+rotation_first_trailer
                if self.first_trailer_rotation > self.maximal_first_trailer_rotation:
                    self.first_trailer_rotation = self.maximal_first_trailer_rotation
                if self.first_trailer_rotation < -self.maximal_first_trailer_rotation:
                    self.first_trailer_rotation = -self.maximal_first_trailer_rotation
                
                step_rotation_first_trailer = self.first_trailer_rotation-old_first_trailer_rotation
                
            if self.number_second_trailers == 2:    
                
                ## second_trailer movement
                first_trailer_hitch_movement = first_trailer_movement                                       +endpoint_movement(hitch_vector_first_trailer_second_trailer,                                                          step_rotation_first_trailer)
                
                second_trailer_movement = vector_rotation(first_trailer_movement,self.second_trailer_rotation)
                
                rotation_second_trailer = np.sign(self.second_trailer_rotation)                                           *angle_two_vectors(hitch_vector_second_trailer                                           -second_trailer_movement                                           +first_trailer_hitch_movement,                                           np.sign(distance)*first_trailer_movement)
                self.second_trailer_rotation = -step_rotation_truck-step_rotation_first_trailer+rotation_second_trailer
                if self.second_trailer_rotation>self.maximal_second_trailer_rotation:
                    self.second_trailer_rotation = self.maximal_second_trailer_rotation
                if self.second_trailer_rotation<-self.maximal_second_trailer_rotation:
                    self.second_trailer_rotation = -self.maximal_second_trailer_rotation
                if self.first_trailer_rotation+self.second_trailer_rotation > self.maximal_both_trailers_rotation:
                    self.second_trailer_rotation = self.maximal_second_trailer_rotation-self.first_trailer_rotation
                if self.first_trailer_rotation+self.second_trailer_rotation < -self.maximal_both_trailers_rotation:
                    self.second_trailer_rotation = -self.maximal_second_trailer_rotation+self.first_trailer_rotation
                    
        return self.truck_translation,self.truck_rotation,self.first_trailer_rotation,self.second_trailer_rotation

