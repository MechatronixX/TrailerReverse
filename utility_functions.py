####################################################################################################
# Utility functions and classes for creating the project environment.                              #
# Contain the functions: out_of_bounds, rectangle_vertices, intersection_area, destination_overlap #
# Contains the classes: Vector, Line                                                               #
####################################################################################################

__author__ = "Pär-Love Palm, Felix Steimle, Jakob Wadman, Veit Wörner"
__copyright__ = "Copyright 2019, Chalmers University of Technology"
__credits__ = ["Pär-Love Palm", "Felix Steimle", "Jakob Wadman", "Veit Wörner"]
__license__ = "GPL"
__version__ = "0.9b"
__maintainer__ = "Felix Steimle, Jakob Wadman"
__email__ = "steimle@student.chalmers.se, wadman@student.chalmers.se"
__status__ = "Production"

import numpy as np
from numpy import array
from numpy.linalg import norm
from math import pi, cos, sin

from vector_rotation_code import *
from constant_rotation_code import *
from endpoint_movement_code import *
from angle_two_vectors_code import *

def out_of_bounds(cog_list, rotation_list, shape_list = [array([7,2]), array([5,2])], yard_shape=array([30,10])):
    '''Checks if the truck or any of the trailer(s) is out of the yard.
        
    Inputs: cog_list  --  List containing the center of gravity for the truck and the trailer(s). The first CoG is the one for the
                          truck, and the following are for trailer1, trailer2, etc. Each CoG is a numpy array of shape (2, )
            rotation_list  --  List containing the rotation for the truck and the trailer(s). The first element is the absolute
                               rotation of the truck, while the remaining elements are the relative rotations of trailer1, trailer2, 
                               etc, wrt the body corresponding to the previous rotation. Each rotation is an angle in degrees.
            shape_list  --  Optional. List containing the shape (dimensions) for the truck and the trailer(s). The first shape is the
                            one for the truck, and the following are for trailer1, trailer2, etc. Each shape is a numpy array of
                            shape (2, )
            yard_shape  --  Optional. Numpy array of shape (2, ) describing the dimensions of the yard. 
            
            
    Output: True if the any object is outside the yard, False if not'''
    
    
    if len(cog_list) != len(rotation_list) or len(cog_list) != len(shape_list):
        raise ValueError('The list of the CoGs, rotation and shapes must have the same lengths // Jakob')
    
    def get_corners_of_rectangle(mid_point, rotation, length, heigth):
        '''Returns the position of the four corners of a rectange based on midpoint and dimentions.
        Inputs: mid_point  --  numpy array of shape (2, )
                rotation  --  roation angle in degrees, scalar
                length  --  x-dimension when rotation=0, scalar
                height  --  y-dimention when rotaion=0, scalar
        Ouput: tuple of the corners, (tr, tl, bl, br), where tr is a numpy array with shape (2, ) that corresponds to the
                rotated top rigth corner, etc.'''
        tr = mid_point + vector_rotation(array([length, heigth])/2, rotation)
        tl = mid_point + vector_rotation(array([-length, heigth])/2, rotation)
        bl = mid_point + vector_rotation(array([-length, -heigth])/2, rotation)
        br = mid_point + vector_rotation(array([length, -heigth])/2, rotation)
        return (tr, tl, bl, br)
    
    def point_out_of_bounds(point):
        '''Checks if the point is outside of the the yard.
        Input - point, numpy array of size (2,)
        Output - True if out of bound, False otherwise'''
        if point[0] > yard_shape[0] or point[0] < 0 or point[1] > yard_shape[1] or point[1] < 0:
            return True
        else:
            return False
    
    # Loop over every body:
    for i in range(len(cog_list)): 
        corners_of_body = get_corners_of_rectangle(cog_list[i], rotation_list[i], shape_list[i][0], shape_list[i][1])
        for corner in corners_of_body:
            if point_out_of_bounds(corner):
                return True
      
    return False

#############################################################################################################################
# Functions and classes for determining the overlapping area of two rectangles.
# From Ruud de Jong on stackoverflow (https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python)

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(cx, cy, w, h, r):
    angle = pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


# Example:
#r1 = (10, 15, 15, 10, 30)
#r2 = (15, 15, 20, 10, 0)
#print(intersection_area(r1, r2))
############################################################################################################################


def destination_overlap(trailer_cog_list, trailer_rotation_list, destination_center, destination_rotation,
                        trailer_shape_list = [array([5, 2])], trailer_config_length = 14):
    '''Checks overlap between destination and the trailers.
        
    Inputs: trailer_cog_list  --  List containing the center of gravity for the trailer(s). The first CoG is the one for the trailer
                                  closest to the truck, and the following go backwards from that. Each CoG is a numpy array of
                                  shape (2, )
            trailer_rotation_list  --  List containing the rotations for the trailer(s). All elements are the ABSOLUTE rotation in
                                       degrees. The first element is the one for the trailer closest to the truck, and the following
                                       go backwards from that.
            destination_center  --  Center of the destination rectangle (CoG if you like), numpy array of shape (2, )
            destination_rotation - rotation angle (degrees) for the destination, scalar
            trailer_shape_list  --  Optional. List containing the shape (dimensions) for the trailer(s). The first shape is the
                            one for the trailer closest to the truck, and the following go backwards from that.  Each shape is a
                            numpy array of shape (2, )
            trailer_config_length  --  Optinal. The length from front of first trailer to rear of last trailer when all rotations
                                       are zero.
            
    Output: Relative overlap between destination and trailers'''
    
    if len(trailer_cog_list) != len(trailer_rotation_list) or len(trailer_cog_list) != len(trailer_shape_list):
        raise ValueError('The list with trailer properties must be the same length // Jakob')
    sum_of_trailer_lengths = sum([dim[0] for dim in trailer_shape_list])
    if trailer_config_length < sum_of_trailer_lengths:
        raise ValueError('The sum of all the trailer length should not be longer that the total trailer configuration // Jakob')
    
    destination_rectangle = (destination_center[0], destination_center[1], trailer_config_length, trailer_shape_list[0][1],
                      destination_rotation)
    # Loop over each trailer:
    overlap_area = 0
    for i in range(len(trailer_cog_list)):
        trailer_rectangle = (trailer_cog_list[i][0], trailer_cog_list[i][1], trailer_shape_list[i][0], trailer_shape_list[i][1],
                            trailer_rotation_list[i])
        overlap_area += intersection_area(trailer_rectangle, destination_rectangle)
        
    
    return overlap_area / (sum_of_trailer_lengths * trailer_shape_list[0][1])


# Example on how to test the destination_overlap function:
#if __name__ == '__main__':
#    destination_center = array([4.5, 5])
#    destination_rotation = 180
#    trailer_cog_list = [array([1.5, 5]), array([7, 5])]
#    trailer_rotation_list = [0, 90]
#    trailer_shape_list = [array([3,2]), array([4,2])]
#    trailer_config_length = 9
#    print(destination_overlap(trailer_cog_list, trailer_rotation_list, destination_center, destination_rotation,
#                        trailer_shape_list = trailer_shape_list, trailer_config_length = trailer_config_length))