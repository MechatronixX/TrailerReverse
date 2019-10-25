#!/usr/bin/env python
# coding: utf-8

###########################################################################
# angle_two_vectors(vector1,vector2)                                      #  
#                                                                         #
# This method is used to measure the relative angle between two vectors   #
# It can handle vectors of length zero, despite that makes no sense       #
# Inputs: vector1 two dimensional vector                                  #
#         vector2 two dimensional vector                                  #
# Outputs: angle relative angle between the two vectors, value in degrees #
###########################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Veit Wörner"
__email__ = "veit@student.chalmers.se"
__status__ = "Production"

import numpy as np
from numpy.linalg import norm

def angle_two_vectors(vector1,vector2):
    if norm(vector1)*norm(vector1) != 0:
        angle = np.rad2deg(np.arccos(np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))))
    else:
        angle = 0
    return angle

