#!/usr/bin/env python
# coding: utf-8

###############################################################################
###############################################################################
###############################################################################
##### Please read carefully!                                              #####
##### For the execution of this program the IDE Spyder is recommended.    #####
##### Please follow the instructions below:                               #####
##### 1. If you use the IDE Spyder, press Strg+F6 on your keyboard while  #####
#####    this code file is opened                                         #####
##### 2. In the category "Console" choose "Execute in an external system  #####
#####    terminal"                                                        #####
##### In case you prefer to use another Python IDE than Spyder with an    #####
##### IPython console, you need to uncomment the following two lines in   #####
##### the code file "Visualize_combination_code":                         #####
##### from IPython import get_ipython                                     #####
##### get_ipython().run_line_magic('matplotlib', 'qt')                    #####
##### Have much fun using this program!                                   #####
###############################################################################
###############################################################################
###############################################################################

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
from multiprocessing import Process, Queue

from visualisation.Visualize_combination_code import *
from simulation.Simulate_combination_code import *
from helperfunctions.load_shapes_code import *

###############################################################################

# The use of multiple processors drastically accelerates simulation and visualization.
# Seldom strange visualizations occur due to a unknown problem.
# This should not influence the simulation.
multicore = True
# Initial state variables
truck_translation = array([np.float(18),np.float(5)])
truck_rotation = 0
first_trailer_rotation = 20
second_trailer_rotation = 20
destination_translation = array([4.5,5])
destination_rotation = 180
number_trailers = 1

# Size of steps
step_size = 1e-2
# Total number of steps
number_steps = 1e5
# Plotting interval in steps
plotting_interval = 1e2

###############################################################################

# Do not change
global plotting_number
plotting_number = 1
pause_time = 0.01

global reading
reading = False

global writing
writing = False

visualisation_shapes = load_shapes(number_trailers)

visualize_combination = Visualize_combination(visualisation_shapes)

simulate_combination = Simulate_combination(visualisation_shapes,\
                                            truck_translation,\
                                            truck_rotation,\
                                            first_trailer_rotation,\
                                            second_trailer_rotation,\
                                            destination_translation,\
                                            destination_rotation,\
                                            number_trailers,\
                                            step_size)
       
def visualize(visualisation_queue):
    while True:
        if visualisation_queue:
            if writing == False:
                global reading
                reading = True
                visualisation_element = visualisation_queue.get()
                reading = False
            if visualisation_element == 'DONE':
                break
            visualize_combination.run(visualisation_element)
            
def simulate(visualisation_queue,velocity_queue,steering_percentage_queue):
    while True:
        if velocity_queue and steering_percentage_queue:
            velocity = velocity_queue.get()
            steering_percentage = steering_percentage_queue.get()
            if velocity == 'DONE' or steering_percentage == 'DONE':
                visualisation_queue.put('DONE')   
                break
            truck_translation,\
            truck_rotation,\
            first_trailer_rotation,\
            second_trailer_rotation = simulate_combination.run(velocity,steering_percentage)
            visualisation_element = [truck_translation, 
                                     truck_rotation,       
                                     first_trailer_rotation,      
                                     second_trailer_rotation,      
                                     steering_percentage,                 
                                     destination_translation,          
                                     destination_rotation,          
                                     number_trailers]
            global plotting_number
            if np.mod(plotting_number,plotting_interval) == 0:
                plotting_number = 1
                if reading == False:
                    global writing
                    writing = True
                    visualisation_queue.put(visualisation_element)
                    writing = False
            else:
                plotting_number += 1
                
def main():
    
    if multicore:
        visualisation_queue = Queue()
        velocity_queue = Queue()
        steering_percentage_queue = Queue()
        
        visualisation_process = Process(target=visualize, args=((visualisation_queue),))
        visualisation_process.daemon = True
        visualisation_process.start()
        
        simulation_process = Process(target=simulate, args=((visualisation_queue),\
                                                            (velocity_queue),\
                                                            (steering_percentage_queue),))
        simulation_process.daemon = True
        simulation_process.start()
        
        for step_number in range(np.int(number_steps)):
            
###############################################################################
########### This is just a demonstration 
            velocity = -np.sin(step_number/1e3*np.pi)
            steering_percentage = np.sin(step_number/1e4*np.pi)
###############################################################################
            
            velocity_queue.put(velocity)
            steering_percentage_queue.put(steering_percentage)
            
        velocity_queue.put('DONE')
        steering_percentage_queue.put('DONE')
        simulation_process.join()
        visualisation_process.join()
    else:        
        for step_number in range(np.int(number_steps)):
            
###############################################################################
########### This is just a demonstration 
            velocity = np.sin(step_number/1e3*np.pi)
            steering_percentage = np.sin(step_number/1e4*np.pi)
###############################################################################
            
            truck_translation,\
            truck_rotation,\
            first_trailer_rotation,\
            second_trailer_rotation = simulate_combination.run(velocity,steering_percentage)
                    
            visualisation_element = [truck_translation, 
                                     truck_rotation,       
                                     first_trailer_rotation,      
                                     second_trailer_rotation,      
                                     steering_percentage,                 
                                     destination_translation,          
                                     destination_rotation,          
                                     number_trailers]
            
            global plotting_number
            if np.mod(plotting_number,plotting_interval) == 0:
                plotting_number = 1
                visualize_combination.run(visualisation_element)
            else:
                plotting_number += 1
    
if __name__=='__main__':
    main()    

