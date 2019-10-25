#!/usr/bin/env python
# coding: utf-8

########################################################################################
# vector_rotation(vector,angle)                                                        #
#                                                                                      #
# This method is used to rotate a two dimensional vector around its origin by an angle #                
# Inputs: vector two dimensional vector                                                #
#         angle  angle, value in degrees                                               #
# Outputs: vector rotated two dimensional vector                                       #
########################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np
from numpy import array

def vector_rotation(vector,angle):
    rotation_matrix = array([[np.cos(np.deg2rad(angle)),-np.sin(np.deg2rad(angle))],
                             [np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))]])
    vector = np.dot(rotation_matrix,vector)
    return vector

