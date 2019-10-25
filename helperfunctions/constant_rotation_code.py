#!/usr/bin/env python
# coding: utf-8

#############################################################################################################################
# constant_rotation(constant,angle)                                                                                         #
#                                                                                                                           #
# This method is used to rotate a two dimensional vector with constant x value and no y value around its origin by an angle #                
# Inputs: constant x value of the rotated vector                                                                            #
#         angle    angle, value in degrees                                                                                  #
# Outputs: vector rotated two dimensional vector                                                                            #
#############################################################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np
from numpy import array

def constant_rotation(constant,angle):
    rotation_vector = array([np.cos(np.deg2rad(angle)),                             np.sin(np.deg2rad(angle))])
    vector = rotation_vector*constant
    return vector

