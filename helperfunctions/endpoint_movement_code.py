#!/usr/bin/env python
# coding: utf-8

################################################################################################################
# endpoint_movement(vector,angle)                                                                              #
#                                                                                                              #
# This method is used to measure the relative movement of the endpoint of a vector during rotation by an angle #                
# Inputs: vector two dimensional vector                                                                        #
#         angle  angle, value in degrees                                                                       #
# Outputs: movement relative movement of the endpoint of the vector                                            #
################################################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np

from helperfunctions.vector_rotation_code import *

def endpoint_movement(vector,angle):
    movement = vector_rotation(vector,angle)-vector
    return movement

