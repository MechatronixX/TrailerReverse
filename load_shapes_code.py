#!/usr/bin/env python
# coding: utf-8

#########################################################################################
# load_shapes(number_trailers)                                                          #
#                                                                                       #
# This method is used to load the shapes of a truck with a variable number of trailers  #                
# Inputs: number_trailers number of trailers, value between 0 and 2                     #                                                                       #
# Outputs: visualisation_shapes array with 28 elements, all of them are described below #
#########################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__copyright__ = "Copyright 2019, Chalmers University of Technology"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

from numpy import array

def load_shapes(number_trailers):
    
    # Shape of the yard
    yard_shape = array([30,10])
    
    # Shape of the destination, here chosen equals to the shape of the second trailer
    destination_shape = array([9,2])
    
    # Shape of some truck, first and second trailer items
    drive_wheel_shape = array([0.8,0.6])
    hitch_radius = 0.1
    
    # Truck item numbers
    # 0: truck
    # 1: cab
    # 2: hitch
    # 3: wheel middle right
    # 4: wheel middle left
    # 5: wheel back right
    # 6: wheel back left
    # 7: wheel front right
    # 8: wheel front left
    
    item_translations_truck = array([
        [0,0],
        [3.75,0],
        [-1.5,0],
        [0,-0.7],
        [0,0.7],
        [-1,-0.8],
        [-1,0.8],
        [3.5,-0.8],
        [3.5,0.8]])
    
    # Shape of the truck items
    truck_shape = array([7,2])
    cab_shape = array([1.5,2])
    wheel_shape = array([0.8,0.4])
    
    # The rotation center of a vehicle lies between the unsteered wheels
    rotation_center_truck = array([2.5,1])
    # This list defines, if and how fast a item rotates relatively, depending on the steering angle
    item_steering_rotations_truck = array([0,0,0,0,0,-0.6,-0.6,1,1])
    # Translation of the hitch relative to the rotation center
    hitch_translation_truck = -1.5
    # Wheelbase of the truck
    wheelbase_truck = 3.5
    # Maximal steering angle of the truck
    maximal_steering_angle = 60
    
    if number_trailers == 1:
        
        # First trailer item numbers
        # 0: chassis
        # 1: shaft
        # 2: wheel back right
        # 3: wheel back left
        # 4: wheel front right
        # 5: wheel front left
        
        item_translations_first_trailer = array([
            [0,0],
            [4,0],
            [-1,-0.8],
            [-1,0.8],
            [0,-0.8],
            [0,0.8]])
        
        # Shape of the first trailer items
        first_trailer_shape = array([5.5,2])
        shaft_shape = array([2,0.2])
        
        # The rotation center of a vehicle lies between the unsteered wheels
        rotation_center_first_trailer = array([2.5,1])
        # This list defines, if and how fast a item rotates relatively, depending on the relative rotation 
        item_steering_rotations_first_trailer = array([0,0,-0.3,-0.3,0,0])
        # Translation of the hitch relative to the rotation center
        hitch_translation_first_trailer_truck = 5
        
    if number_trailers == 2:
        
        # First trailer item numbers
        # 0: chassis
        # 1: shaft
        # 2: hitch
        # 3: wheel back right
        # 4: wheel back left
        # 5: wheel front right
        # 6: wheel front left
        
        item_translations_first_trailer = array([
            [0,0],
            [1.5,0],
            [-0.5,0],
            [-1,-0.8],
            [-1,0.8],
            [0,-0.8],
            [0,0.8]])
        
        # Shape of the first trailer items
        first_trailer_shape = array([2,2])
        shaft_shape = array([2,0.2])
        
        # The rotation center of a vehicle lies between the unsteered wheels
        rotation_center_first_trailer = array([1.5,1])
        # This list defines, if and how fast a item rotates relatively, depending on the relative rotation
        item_steering_rotations_first_trailer = array([0,0,0,-0.4,-0.4,0,0])
        # Translation of the hitchs relative to the rotation center
        hitch_translation_first_trailer_truck = 2.5
        hitch_translation_first_trailer_second_trailer = -0.5
        
        # Second trailer item numbers
        # 0: chassis
        # 1: shaft
        # 2: wheel back right
        # 3: wheel back left
        # 4: wheel front right
        # 5: wheel front left
        
        item_translations_second_trailer = array([
            [0,0],
            [-1,-0.8],
            [-1,0.8],
            [0,-0.8],
            [0,0.8],
            [1,-0.8],
            [1,0.8]])
        
        # Shape of the first trailer items
        second_trailer_shape = array([9,2])
        
        # The rotation center of a vehicle lies between the unsteered wheels
        rotation_center_second_trailer = array([2.5,1])
        # This list defines, if and how fast a item rotates relatively, depending on the relative rotation
        item_steering_rotations_second_trailer = array([0,-0.2,-0.2,0,0,0.2,0.2])
        # Translation of the hitch relative to the rotation center
        hitch_translation_second_trailer = 5.5
        
    maximal_first_trailer_rotation = 90
    maximal_second_trailer_rotation = 180
    maximal_both_trailers_rotation = 200
        
    visualisation_shapes = [yard_shape,\
                        destination_shape,\
                        drive_wheel_shape,\
                        hitch_radius,\
                        item_translations_truck,\
                        truck_shape,\
                        cab_shape,\
                        wheel_shape,\
                        rotation_center_truck,\
                        item_steering_rotations_truck,\
                        hitch_translation_truck,\
                        wheelbase_truck,\
                        maximal_steering_angle,\
                        item_translations_first_trailer,\
                        first_trailer_shape,\
                        shaft_shape,\
                        rotation_center_first_trailer,\
                        item_steering_rotations_first_trailer,\
                        hitch_translation_first_trailer_truck,\
                        hitch_translation_first_trailer_second_trailer,\
                        item_translations_second_trailer,\
                        second_trailer_shape,\
                        rotation_center_second_trailer,\
                        item_steering_rotations_second_trailer,\
                        hitch_translation_second_trailer,\
                        maximal_first_trailer_rotation,\
                        maximal_second_trailer_rotation,\
                        maximal_both_trailers_rotation]

    return visualisation_shapes
