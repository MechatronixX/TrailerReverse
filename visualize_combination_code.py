#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python

##################################################################################################################
# visualize_combination(visualisation_element)                                                                   #
#                                                                                                                #
# This method is used to plot the combination consisting of a truck and possible second_trailers                 #
# Inputs: visualisation_element   Array with the following 8 elements:                                           #
#                                                                                                                #
#         truck_translation       absolute translation of the truck                                              #
#         truck_rotation          absolute rotation of the truck, value in degrees                               #
#         first_trailer_rotation  relative rotation of the first trailer to the truck, value in degrees          #
#         second_trailer_rotation relative rotation of the second trailer to the first trailer, value in degrees #
#         steering_percentage     steering angle of the truck, value between -1 and 1                            #
#         destination_translation absolute translation of the destination                                        #
#         destination_rotation    absolute rotation of the destination, value in degrees                         #
#         number_trailers         number of trailers, value between 0 and 2                                      #
# Outputs: _                                                                                                     #
##################################################################################################################

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import transforms
import numpy as np
from numpy import array
get_ipython().run_line_magic('matplotlib', 'inline')

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__copyright__ = "Copyright 2019, Chalmers University of Technology"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

def visualize_combination(visualisation_element):
    
    truck_translation,    truck_rotation,    first_trailer_rotation,    second_trailer_rotation,    steering_percentage,    destination_translation,    destination_rotation,    number_second_trailers = visualisation_element
    
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
    
    maximal_steering_angle = 60
    
    if number_second_trailers == 1:
        
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
        
    if number_second_trailers == 2:
        
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
    
############################################################################################################################
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111, aspect='equal')
    
    # Hiding the axes
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    
    ## Destination plotting
    destination_rectangle = patches.Rectangle(-destination_shape/2,                                              destination_shape[0],                                              destination_shape[1],                                              color="red",                                              alpha=0.50)
    rotation_destination = transforms.Affine2D().rotate_deg(destination_rotation)
    translation_destination = transforms.Affine2D().translate(destination_translation[0],                                                              destination_translation[1])
    destination_transformation = rotation_destination                                 +translation_destination                                 +ax1.transData
    destination_rectangle.set_transform(destination_transformation)
    ax1.add_patch(destination_rectangle)
    
    ## Truck plotting
    wheel_rectangles = []
    everything = []
    truck_rectangle = patches.Rectangle(-rotation_center_truck,                                        truck_shape[0],                                        truck_shape[1],                                        color="black",                                        alpha=0.50)
    cab_rectangle = patches.Rectangle(-cab_shape/2,                                      cab_shape[0],                                      cab_shape[1],                                      color="green",                                      alpha=0.50)
    drive_wheel_rectangles = [patches.Rectangle(-drive_wheel_shape/2,                                          drive_wheel_shape[0],                                          drive_wheel_shape[1],                                          color="black",                                          alpha=0.50)                                          for drive_wheel_number in range(2)]
    wheel_rectangles = [patches.Rectangle(-wheel_shape/2,                                          wheel_shape[0],                                          wheel_shape[1],                                          color="black",                                          alpha=0.50)                                          for wheel_number in range(5)]
    hitch_circle = patches.Circle((0, 0),                                  hitch_radius,                                  color="black",                                  alpha=0.50)
    everything = [truck_rectangle]+[cab_rectangle]+[hitch_circle]+drive_wheel_rectangles+wheel_rectangles
    
    item_number = 0
    item_rotation = []
    item_translation = []
    for item_number in range(len(everything)-1):
        # rotate wheels
        item_rotation = item_rotation                        +[transforms.Affine2D().rotate_deg(steering_percentage                                                           *maximal_steering_angle                                                           *item_steering_rotations_truck[item_number])]
        # translate wheels
        item_translation = item_translation                           +[transforms.Affine2D().translate(item_translations_truck[item_number,0],                                                             item_translations_truck[item_number,1])]
    # rotate truck with wheels
    rotation_everything = transforms.Affine2D().rotate_deg(truck_rotation)
    # translate truck with wheels
    translation_everything = transforms.Affine2D().translate(truck_translation[0],                                                             truck_translation[1])
    
    for item_number in range(len(everything)-1):
        item_transformation = item_rotation[item_number]                              +item_translation[item_number]                              +rotation_everything                              +translation_everything                              +ax1.transData
        everything[item_number].set_transform(item_transformation)
        ax1.add_patch(everything[item_number])
    
    ## first_trailer plotting
    if number_second_trailers != 0:
        wheel_rectangles = []
        everything = []
        first_trailer_rectangle = patches.Rectangle(-rotation_center_first_trailer,                                            first_trailer_shape[0],                                            first_trailer_shape[1],                                            color="black",                                            alpha=0.50)
        shaft_rectangle = patches.Rectangle(-shaft_shape/2,                                            shaft_shape[0],                                            shaft_shape[1],                                            color="black",                                            alpha=0.50)
        wheel_rectangles = [patches.Rectangle(-wheel_shape/2,                                              wheel_shape[0],                                              wheel_shape[1],                                              color="black",                                              alpha=0.50)                                              for wheel_number in range(5)]
     
        if number_second_trailers == 1:
            everything = [first_trailer_rectangle]+[shaft_rectangle]+wheel_rectangles
         
        if number_second_trailers == 2:
            hitch_circle = patches.Circle((0, 0),                                          hitch_radius,                                          color="black",                                          alpha=0.50)
            everything = [first_trailer_rectangle]+[shaft_rectangle]+[hitch_circle]+wheel_rectangles
        
        item_number = 0
        item_rotation = []
        item_translation = []
        for item_number in range(len(everything)-1):
            # rotate wheels
            item_rotation = item_rotation                            +[transforms.Affine2D().rotate_deg(-first_trailer_rotation                                                               *item_steering_rotations_first_trailer[item_number])]
            # translate wheels
            item_translation = item_translation                               +[transforms.Affine2D().translate(item_translations_first_trailer[item_number,0],                                                                 item_translations_first_trailer[item_number,1])]
        # rotate first_trailer with wheels
        rotation_everything = transforms.Affine2D().rotate_deg(truck_rotation                                                               +first_trailer_rotation)
        # translate first_trailer with wheels
        translation_everything = transforms.Affine2D().translate(truck_translation[0]                                                                 +hitch_translation_truck                                                                 *np.cos(np.deg2rad(truck_rotation))                                                                 -hitch_translation_first_trailer_truck                                                                 *np.cos(np.deg2rad(truck_rotation+first_trailer_rotation)),                                                                 truck_translation[1]                                                                 +hitch_translation_truck                                                                 *np.sin(np.deg2rad(truck_rotation))                                                                 -hitch_translation_first_trailer_truck                                                                 *np.sin(np.deg2rad(truck_rotation+first_trailer_rotation)))
        
        for item_number in range(len(everything)-1):
            item_transformation = item_rotation[item_number]                                  +item_translation[item_number]                                  +rotation_everything                                  +translation_everything                                  +ax1.transData
            everything[item_number].set_transform(item_transformation)
            ax1.add_patch(everything[item_number])
           
    ## second_trailer plotting    
    if number_second_trailers == 2:
        wheel_rectangles = []
        everything = []
        second_trailer_rectangle = patches.Rectangle(-rotation_center_second_trailer,                                              second_trailer_shape[0],                                              second_trailer_shape[1],                                              color="black",                                              alpha=0.50)
        wheel_rectangles = [patches.Rectangle(-wheel_shape/2,                                              wheel_shape[0],                                              wheel_shape[1],                                              color="black",                                              alpha=0.50)                                              for wheel_number in range(7)]
        everything = [second_trailer_rectangle]+wheel_rectangles
        
        item_number = 0
        item_rotation = []
        item_translation = []
        for item_number in range(len(everything)-1):
            # rotate wheels
            item_rotation = item_rotation                            +[transforms.Affine2D().rotate_deg(-second_trailer_rotation                                                               *item_steering_rotations_second_trailer[item_number])]
            # translate wheels
            item_translation = item_translation                               +[transforms.Affine2D().translate(item_translations_second_trailer[item_number,0],                                                                 item_translations_second_trailer[item_number,1])]
        # rotate second_trailer with wheels
        rotation_everything = transforms.Affine2D().rotate_deg(truck_rotation                                                               +first_trailer_rotation                                                               +second_trailer_rotation)
        # translate second_trailer with wheels
        translation_everything = transforms.Affine2D().translate(truck_translation[0]                                                                 +hitch_translation_truck                                                                 *np.cos(np.deg2rad(truck_rotation))                                                                 +(-hitch_translation_first_trailer_truck+hitch_translation_first_trailer_second_trailer)                                                                 *np.cos(np.deg2rad(truck_rotation+first_trailer_rotation))                                                                 -hitch_translation_second_trailer                                                                 *np.cos(np.deg2rad(truck_rotation+first_trailer_rotation+second_trailer_rotation)),                                                                 truck_translation[1]                                                                 +hitch_translation_truck                                                                 *np.sin(np.deg2rad(truck_rotation))                                                                 +(-hitch_translation_first_trailer_truck+hitch_translation_first_trailer_second_trailer)                                                                 *np.sin(np.deg2rad(truck_rotation+first_trailer_rotation))                                                                 -hitch_translation_second_trailer                                                                 *np.sin(np.deg2rad(truck_rotation+first_trailer_rotation+second_trailer_rotation)))
        
        for item_number in range(len(everything)-1):
            item_transformation = item_rotation[item_number]                                  +item_translation[item_number]                                  +rotation_everything                                  +translation_everything                                  +ax1.transData
            everything[item_number].set_transform(item_transformation)
            ax1.add_patch(everything[item_number])
    
    plt.xlim(0, yard_shape[0])
    plt.ylim(0, yard_shape[1])
    plt.show()

