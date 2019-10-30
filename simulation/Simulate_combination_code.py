#!/usr/bin/env python
# coding: utf-8

##############################################################################################################################
# Simulate_combination()                                                                                                     #    
#                                                                                                                            #
# This class initialises the simulation of of a truck with a variable number of trailers                                     #                 
# Inputs: visualisation_shapes array provided by the method load_shapes(number_trailers)                                     #
#         number_trailers      number of trailers, value between 0 and 2                                                     #
#         step_size            size of the time steps                                                                        #
# Outputs: _                                                                                                                 #
# Methods: run(truck_translation,truck_rotation,first_trailer_rotation,second_trailer_rotation,velocity,steering_percentage) #
#                                                                                                                            #
# run(truck_translation,truck_rotation,first_trailer_rotation,second_trailer_rotation,velocity,steering_percentage)          #                                                                               #
# This method is used to simulate the movement of a truck with a variable number of trailers                                 #
# Inputs: truck_translation       absolute translation of the truck                                                          #
#         truck_rotation          absolute rotation of the truck, value in degrees                                           #
#         first_trailer_rotation  relative rotation of the first trailer to the truck, value in degrees                      #
#         second_trailer_rotation relative rotation of the second trailer to the first trailer, value in degrees             #
#         velocity                velocity of the truck, value between -1 and 1                                              #
#         steering_percentage     steering angle of the truck, value between -1 and 1                                        #
# Outputs: truck_translation       absolute translation of the truck                                                         #
#          truck_rotation          absolute rotation of the truck, value in degrees                                          #
#          first_trailer_rotation  relative rotation of the first trailer to the truck, value in degrees                     #
#          second_trailer_rotation relative rotation of the second trailer to the first trailer, value in degrees            #
##############################################################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np
from numpy import array
import copy 

from helperfunctions.vector_rotation_code import *
from helperfunctions.constant_rotation_code import *
from helperfunctions.endpoint_movement_code import *
from helperfunctions.angle_two_vectors_code import *

class Simulate_combination():
    def __init__(self,\
                 visualisation_shapes,\
                 number_trailers,\
                 step_size):

        self.yard_shape,\
        self.drive_wheel_shape,\
        self.hitch_radius,\
        self.item_translations_truck,\
        self.truck_shape,\
        self.cab_shape,\
        self.wheel_shape,\
        self.rotation_center_truck,\
        self.item_steering_rotations_truck,\
        self.hitch_translation_truck,\
        self.wheelbase_truck,\
        self.maximal_steering_angle,\
        self.item_translations_first_trailer,\
        self.first_trailer_shape,\
        self.shaft_shape,\
        self.rotation_center_first_trailer,\
        self.item_steering_rotations_first_trailer,\
        self.hitch_translation_first_trailer_truck,\
        self.hitch_translation_first_trailer_second_trailer,\
        self.item_translations_second_trailer,\
        self.second_trailer_shape,\
        self.rotation_center_second_trailer,\
        self.item_steering_rotations_second_trailer,\
        self.hitch_translation_second_trailer,\
        self.maximal_first_trailer_rotation,\
        self.maximal_second_trailer_rotation,\
        self.maximal_both_trailers_rotation= visualisation_shapes
        
        self.number_trailers = number_trailers
        self.step_size = step_size        

    def run(self,truck_translation,truck_rotation,first_trailer_rotation,second_trailer_rotation,velocity,steering_percentage):
        steering_angle = steering_percentage*self.maximal_steering_angle
        
        #This step basically seems to perform euler forward for the truck. 
        distance = velocity*self.step_size
        
        truck_movement = constant_rotation(distance,truck_rotation)
        first_trailer_movement = vector_rotation(truck_movement,first_trailer_rotation)
        second_trailer_movement = vector_rotation(first_trailer_movement,second_trailer_rotation)
        
        hitch_vector_truck = constant_rotation(self.hitch_translation_truck,\
                                               truck_rotation)
        hitch_vector_first_trailer_truck = constant_rotation(self.hitch_translation_first_trailer_truck,\
                                                             truck_rotation+\
                                                             first_trailer_rotation)
        hitch_vector_first_trailer_second_trailer = constant_rotation(self.hitch_translation_first_trailer_second_trailer,\
                                                                      truck_rotation+\
                                                                      first_trailer_rotation)
        hitch_vector_second_trailer = constant_rotation(self.hitch_translation_second_trailer,\
                                                        truck_rotation+\
                                                        first_trailer_rotation+\
                                                        second_trailer_rotation)
        
        if norm(truck_movement) != 0:
            
            old_truck_rotation = truck_rotation
            
            rotation_truck = 0      
            # truck movement
            if steering_percentage != 0:
                lock = self.wheelbase_truck/np.tan(np.deg2rad(steering_angle))
                rotation_truck = np.rad2deg(np.arcsin(distance/lock))/2
            truck_rotation += rotation_truck
            
            translation_truck = vector_rotation(truck_movement,rotation_truck)
            truck_translation += translation_truck
          
            step_rotation_truck = truck_rotation-old_truck_rotation
            old_first_trailer_rotation = first_trailer_rotation
            
            if self.number_trailers >= 1:
                # first_trailer movement
                truck_hitch_movement = truck_movement\
                                       +endpoint_movement(hitch_vector_truck,\
                                                          step_rotation_truck)
                
                first_trailer_movement = vector_rotation(truck_movement,first_trailer_rotation)
                
                rotation_first_trailer = np.sign(first_trailer_rotation)\
                                         *angle_two_vectors(hitch_vector_first_trailer_truck\
                                                            -first_trailer_movement\
                                                            +truck_hitch_movement,\
                                                            np.sign(distance)*truck_movement)
                    
                first_trailer_rotation = -step_rotation_truck+rotation_first_trailer
                if first_trailer_rotation > self.maximal_first_trailer_rotation:
                    first_trailer_rotation = self.maximal_first_trailer_rotation
                if first_trailer_rotation < -self.maximal_first_trailer_rotation:
                    first_trailer_rotation = -self.maximal_first_trailer_rotation
                
                step_rotation_first_trailer = first_trailer_rotation-old_first_trailer_rotation
                
            if self.number_trailers == 2:    
                # second_trailer movement
                first_trailer_hitch_movement = first_trailer_movement\
                                       +endpoint_movement(hitch_vector_first_trailer_second_trailer,\
                                                          step_rotation_first_trailer)
                
                second_trailer_movement = vector_rotation(first_trailer_movement,second_trailer_rotation)
                
                rotation_second_trailer = np.sign(second_trailer_rotation)\
                                           *angle_two_vectors(hitch_vector_second_trailer\
                                                              -second_trailer_movement\
                                                              +first_trailer_hitch_movement,\
                                                              np.sign(distance)*first_trailer_movement)
                                           
                if np.isnan(angle_two_vectors(hitch_vector_second_trailer\
                                                              -second_trailer_movement\
                                                              +first_trailer_hitch_movement,\
                                                              np.sign(distance)*first_trailer_movement)):
                    print(hitch_vector_second_trailer\
                                                              -second_trailer_movement\
                                                              +first_trailer_hitch_movement)
                    print(np.sign(distance)*first_trailer_movement)
                    wait(1000)
                    
                second_trailer_rotation = -step_rotation_truck-step_rotation_first_trailer+rotation_second_trailer
                if second_trailer_rotation>self.maximal_second_trailer_rotation:
                    second_trailer_rotation = self.maximal_second_trailer_rotation
                if second_trailer_rotation<-self.maximal_second_trailer_rotation:
                    second_trailer_rotation = -self.maximal_second_trailer_rotation
                if first_trailer_rotation+second_trailer_rotation > self.maximal_both_trailers_rotation:
                    second_trailer_rotation = self.maximal_second_trailer_rotation-first_trailer_rotation
                if first_trailer_rotation+second_trailer_rotation < -self.maximal_both_trailers_rotation:
                    second_trailer_rotation = -self.maximal_second_trailer_rotation+first_trailer_rotation
                    
        return truck_translation,truck_rotation,first_trailer_rotation,second_trailer_rotation

