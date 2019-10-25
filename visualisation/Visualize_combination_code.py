#!/usr/bin/env python
# coding: utf-8

##################################################################################################################
# Visualize_combination()                                                                                        #
#                                                                                                                #
# This class initialises the visualisation of of a truck with a variable number of trailers                      #
# Inputs: visualisation_shapes array provided by the method load_shapes(number_trailers)                         #
# Outputs: _                                                                                                     #
# Methods: run(visualisation_element)                                                                            #
#                                                                                                                #
# run(visualisation_element)                                                                                     #
# This method is used to plot the combination consisting of a truck and possible trailers                        #
# Inputs: visualisation_element   array with the following 8 elements:                                           #
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

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import transforms
import numpy as np
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

class Visualize_combination():
    def __init__(self,visualisation_shapes):
        
        self.yard_shape,\
        self.destination_shape,\
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
            
        plt.ion()
        visualisation_figure = plt.figure(figsize=(10,10))
        self.ax = visualisation_figure.add_subplot(111, aspect='equal')
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        plt.xlim(0, self.yard_shape[0])
        plt.ylim(0, self.yard_shape[1])
        plt.show()
        
        self.pause_time = 0.01

    def run(self,visualisation_element):
        
        for old_element in reversed(self.ax.patches):
            old_element.remove()
        
        truck_translation,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        steering_percentage,\
        destination_translation,\
        destination_rotation,\
        number_trailers = visualisation_element
        
        ## Destination plotting
        destination_rectangle = patches.Rectangle(-self.destination_shape/2,\
                                                  self.destination_shape[0],\
                                                  self.destination_shape[1],\
                                                  color="red",\
                                                  alpha=0.50)
        rotation_destination = transforms.Affine2D().rotate_deg(destination_rotation)
        translation_destination = transforms.Affine2D().translate(destination_translation[0],\
                                                                  destination_translation[1])
        destination_transformation = rotation_destination\
              +translation_destination\
              +self.ax.transData
        destination_rectangle.set_transform(destination_transformation)
        self.ax.add_patch(destination_rectangle)
        
        ## Truck plotting
        wheel_rectangles = []
        truck_everything = []
        truck_rectangle = patches.Rectangle(-self.rotation_center_truck,\
                                            self.truck_shape[0],\
                                            self.truck_shape[1],\
                                            color="black",\
                                            alpha=0.50)
        cab_rectangle = patches.Rectangle(-self.cab_shape/2,\
                                          self.cab_shape[0],\
                                          self.cab_shape[1],\
                                          color="green",\
                                          alpha=0.50)
        drive_wheel_rectangles = [patches.Rectangle(-self.drive_wheel_shape/2,\
                                                    self.drive_wheel_shape[0],\
                                                    self.drive_wheel_shape[1],\
                                                    color="black",\
                                                    alpha=0.50)\
            for drive_wheel_number in range(2)]
        wheel_rectangles = [patches.Rectangle(-self.wheel_shape/2,\
                                              self.wheel_shape[0],\
                                              self.wheel_shape[1],\
                                              color="black",\
                                              alpha=0.50)\
                      for wheel_number in range(5)]
        hitch_circle = patches.Circle((0, 0),\
                                      self.hitch_radius,\
                                      color="black",\
                                      alpha=0.50)
        truck_everything = [truck_rectangle]\
            +[cab_rectangle]\
            +[hitch_circle]\
            +drive_wheel_rectangles\
            +wheel_rectangles
        
        item_number = 0
        item_rotation = []
        item_translation = []
        for item_number in range(len(truck_everything)-1):
            # rotate wheels
            item_rotation = item_rotation\
                +[transforms.Affine2D().rotate_deg(steering_percentage\
                                                   *self.maximal_steering_angle\
                                                   *self.item_steering_rotations_truck[item_number])]
            # translate wheels
            item_translation = item_translation\
                 +[transforms.Affine2D().translate(self.item_translations_truck[item_number,0],\
                                                            self.item_translations_truck[item_number,1])]
        # rotate truck with wheels
        rotation_everything = transforms.Affine2D().rotate_deg(truck_rotation)
        # translate truck with wheels
        translation_everything = transforms.Affine2D().translate(truck_translation[0],\
                                                             truck_translation[1])
        
        for item_number in range(len(truck_everything)-1):
            item_transformation = item_rotation[item_number]\
               +item_translation[item_number]\
               +rotation_everything\
               +translation_everything\
               +self.ax.transData
            truck_everything[item_number].set_transform(item_transformation)
            self.ax.add_patch(truck_everything[item_number])
        
        ## first_trailer plotting
        if number_trailers != 0:
            wheel_rectangles = []
            first_trailer_everything = []
            first_trailer_rectangle = patches.Rectangle(-self.rotation_center_first_trailer,\
                                                        self.first_trailer_shape[0],\
                                                        self.first_trailer_shape[1],\
                                                        color="black",\
                                                        alpha=0.50)
            shaft_rectangle = patches.Rectangle(-self.shaft_shape/2,\
                                                self.shaft_shape[0],\
                                                self.shaft_shape[1],\
                                                color="black",\
                                                alpha=0.50)
            wheel_rectangles = [patches.Rectangle(-self.wheel_shape/2,\
                                                  self.wheel_shape[0],\
                                                  self.wheel_shape[1],\
                                                  color="black",\
                                                  alpha=0.50)\
                         for wheel_number in range(5)]
         
            if number_trailers == 1:
                first_trailer_everything = [first_trailer_rectangle]\
                    +[shaft_rectangle]\
                    +wheel_rectangles
             
            if number_trailers == 2:
                hitch_circle = patches.Circle((0, 0),\
                                              self.hitch_radius,\
                                              color="black",\
                                              alpha=0.50)
                first_trailer_everything = [first_trailer_rectangle]\
                    +[shaft_rectangle]\
                    +[hitch_circle]\
                    +wheel_rectangles
            
            item_number = 0
            item_rotation = []
            item_translation = []
            for item_number in range(len(first_trailer_everything)-1):
                # rotate wheels
                item_rotation = item_rotation\
                    +[transforms.Affine2D().rotate_deg(-first_trailer_rotation\
                    *self.item_steering_rotations_first_trailer[item_number])]
                # translate wheels
                item_translation = item_translation\
                    +[transforms.Affine2D().translate(self.item_translations_first_trailer[item_number,0],\
                                                                self.item_translations_first_trailer[item_number,1])]
            # rotate first_trailer with wheels
            rotation_everything = transforms.Affine2D().rotate_deg(truck_rotation\
                                                         +first_trailer_rotation)
            # translate first_trailer with wheels
            translation_everything = transforms.Affine2D().translate(truck_translation[0]\
                                                                     +self.hitch_translation_truck\
                                                                     *np.cos(np.deg2rad(truck_rotation))\
                                                                     -self.hitch_translation_first_trailer_truck\
                                                                     *np.cos(np.deg2rad(truck_rotation\
                                                                                        +first_trailer_rotation)),\
                                                                     truck_translation[1]\
                                                                     +self.hitch_translation_truck\
                                                                     *np.sin(np.deg2rad(truck_rotation))\
                                                                     -self.hitch_translation_first_trailer_truck\
                                                                     *np.sin(np.deg2rad(truck_rotation\
                                                                                        +first_trailer_rotation)))
            
            for item_number in range(len(first_trailer_everything)-1):
                item_transformation = item_rotation[item_number]\
                    +item_translation[item_number]\
                    +rotation_everything\
                    +translation_everything\
                    +self.ax.transData
                first_trailer_everything[item_number].set_transform(item_transformation)
                self.ax.add_patch(first_trailer_everything[item_number])
               
        ## second_trailer plotting    
        if number_trailers == 2:
            wheel_rectangles = []
            second_trailer_everything = []
            second_trailer_rectangle = patches.Rectangle(-self.rotation_center_second_trailer,\
                                                         self.second_trailer_shape[0],\
                                                         self.second_trailer_shape[1],\
                                                         color="black",\
                                                         alpha=0.50)
            wheel_rectangles = [patches.Rectangle(-self.wheel_shape/2,\
                                                  self.wheel_shape[0],\
                                                  self.wheel_shape[1],\
                                                  color="black",\
                                                  alpha=0.50)\
                       for wheel_number in range(7)]
            second_trailer_everything = [second_trailer_rectangle]\
                +wheel_rectangles
            
            item_number = 0
            item_rotation = []
            item_translation = []
            for item_number in range(len(second_trailer_everything)-1):
                # rotate wheels
                item_rotation = item_rotation\
                    +[transforms.Affine2D().rotate_deg(-second_trailer_rotation\
                    *self.item_steering_rotations_second_trailer[item_number])]
                # translate wheels
                item_translation = item_translation\
                    +[transforms.Affine2D().translate(self.item_translations_second_trailer[item_number,0],\
                                                      self.item_translations_second_trailer[item_number,1])]
            # rotate second_trailer with wheels
            rotation_everything = transforms.Affine2D().rotate_deg(truck_rotation\
                                                                   +first_trailer_rotation\
                                                                   +second_trailer_rotation)
            # translate second_trailer with wheels
            translation_everything = transforms.Affine2D().translate(truck_translation[0]\
                                                             +self.hitch_translation_truck\
                                                               *np.cos(np.deg2rad(truck_rotation))\
                                                               +(-self.hitch_translation_first_trailer_truck\
                                                                 +self.hitch_translation_first_trailer_second_trailer)\
                                                                 *np.cos(np.deg2rad(truck_rotation\
                                                                                    +first_trailer_rotation))\
                                                                 -self.hitch_translation_second_trailer\
                                                                 *np.cos(np.deg2rad(truck_rotation\
                                                                                    +first_trailer_rotation\
                                                                                    +second_trailer_rotation)),\
                                                                                    truck_translation[1]\
                                                                                    +self.hitch_translation_truck\
                                                                                    *np.sin(np.deg2rad(truck_rotation))\
                                                                                    +(-self.hitch_translation_first_trailer_truck\
                                                                                    +self.hitch_translation_first_trailer_second_trailer)\
                                                                                    *np.sin(np.deg2rad(truck_rotation\
                                                                                                       +first_trailer_rotation))\
                                                                                    -self.hitch_translation_second_trailer\
                                                                                    *np.sin(np.deg2rad(truck_rotation\
                                                                                                       +first_trailer_rotation\
                                                                                                       +second_trailer_rotation)))
            
            for item_number in range(len(second_trailer_everything)-1):
                item_transformation = item_rotation[item_number]\
                    +item_translation[item_number]\
                    +rotation_everything\
                    +translation_everything\
                    +self.ax.transData
                second_trailer_everything[item_number].set_transform(item_transformation)
                self.ax.add_patch(second_trailer_everything[item_number])

        plt.pause(self.pause_time)
