import gym
import numpy as np
from gym import spaces

from visualisation.Visualize_combination_code import *
from simulation.Simulate_combination_code import *
from helperfunctions.load_shapes_code import *

class Reverse_variable_trailer_number_environment(gym.Env):
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 25}

    def __init__(self):
        
        self.name = "Reverse_variable_trailer_number"
        
        self.destination_translation = np.array([2.5,2.5])
        self.destination_rotation = 0
        self.number_trailers = 2
        
        visualisation_shapes = load_shapes(self.number_trailers)
        
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
        
        self.maximal_velocity = 1
        self.maximal_steering_percentage = 1
        
        high = np.array([self.yard_shape[0],\
                         self.yard_shape[1],\
                         360,\
                         360,\
                         360,\
                         self.maximal_velocity,\
                         self.maximal_steering_percentage])
        
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.viewer = None

        # Size of steps
        self.step_size = 1e-2
        # Plotting interval in steps
        self.plotting_number = 1
        self.plotting_interval = 1e1
        
        self.velocity_increment = 0.1
        self.steering_percentage_increment = 0.1
        
        self.jackknife_angle = 90
        self.jackknive_number = 0
        
        self.visualisation = False
        
        self.velocity_old = 0
        self.velocity_new = 0
        
        #Watchdog, so the vehicle doesnt wander around for too long, missing the target. 
        self.traveled_distance = 0
        self.maximal_traveled_distance = 0
        
        self.state = None
        
        #Reset to the initial state. 
        self.reset()
        
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        if self.visualisation:
            self.visualize_combination = Visualize_combination(visualisation_shapes)
        
        self.simulate_combination = Simulate_combination(visualisation_shapes,\
                                                         self.number_trailers,\
                                                         self.step_size)
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        truck_translation = np.array([truck_translation_x,truck_translation_y])
        
        truck_translation,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation = self.simulate_combination.run(truck_translation,\
                                                                truck_rotation,\
                                                                first_trailer_rotation,\
                                                                second_trailer_rotation,\
                                                                velocity,\
                                                                steering_percentage)
        self.velocity_old = velocity
        
        if action == 0: 
            None
        elif action == 1: 
            velocity -= self.velocity_increment
        elif action == 2:
            velocity += self.velocity_increment
        elif action == 3: 
            steering_percentage += self.steering_percentage_increment
        elif action == 4: 
            steering_percentage -= self.steering_percentage_increment
            
        if (self.check_jackknife() and velocity <= 0 ): 
            velocity = 0 
            
        #Clamp velocity within reasonanle boundaries.     
        velocity = np.clip(velocity,-self.maximal_velocity,self.maximal_velocity)
        
        self.velocity_new = velocity
        
        self.traveled_distance += velocity*self.step_size
        
        steering_percentage = np.clip(steering_percentage,-self.maximal_steering_percentage,self.maximal_steering_percentage)
       
        self.state = np.array([truck_translation[0],\
                               truck_translation[1],\
                               truck_rotation,\
                               first_trailer_rotation,\
                               second_trailer_rotation,\
                               velocity,\
                               steering_percentage])
        
        reward, done = self.calc_reward()
        
        if self.visualisation:          
            self.visualize()
            
        return self.state, reward, done, {}
    
    def check_jackknife(self): 
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        jackknive = np.abs(first_trailer_rotation) >= self.jackknife_angle or\
                           np.abs(second_trailer_rotation) >= self.jackknife_angle
      
        return jackknive
    
    def check_out_of_bounds(self): 
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        truck_absolute_translation,\
        truck_absolute_rotation,\
        first_trailer_absolute_translation,\
        first_trailer_absolute_rotation,\
        second_trailer_absolute_translation,\
        second_trailer_absolute_rotation = self.get_absolute_translation_and_rotation_of_truck_and_trailer()
        
        out_of_bounds = (truck_absolute_translation[0] > self.yard_shape[0]) or\
                        (truck_absolute_translation[0] < 0) or\
                        (truck_absolute_translation[1] > self.yard_shape[1]) or\
                        (truck_absolute_translation[1] < 0) or\
                        (first_trailer_absolute_translation[0] > self.yard_shape[0]) or\
                        (first_trailer_absolute_translation[0] < 0) or\
                        (first_trailer_absolute_translation[1] > self.yard_shape[1]) or\
                        (first_trailer_absolute_translation[1] < 0) or\
                        (second_trailer_absolute_translation[0] > self.yard_shape[0]) or\
                        (second_trailer_absolute_translation[0] < 0) or\
                        (second_trailer_absolute_translation[1] > self.yard_shape[1]) or\
                        (second_trailer_absolute_translation[1] < 0)
                        
        return out_of_bounds
    
    def calc_reward(self):
        done = False
        
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        reward = 0
        
        truck_absolute_translation,\
        truck_absolute_rotation,\
        first_trailer_absolute_translation,\
        first_trailer_absolute_rotation,\
        second_trailer_absolute_translation,\
        second_trailer_absolute_rotation = self.get_absolute_translation_and_rotation_of_truck_and_trailer()
        
        if self.number_trailers == 0:
            translation_error = np.linalg.norm(truck_absolute_translation-self.destination_translation)
        elif self.number_trailers == 2:
            translation_error = np.linalg.norm(second_trailer_absolute_translation-self.destination_translation)
        else:
            translation_error = np.linalg.norm(first_trailer_absolute_translation-self.destination_translation)
        
        # The reward is the lower, the higher the distance to the target is 
        reward = -1e3*(translation_error)**2
        
        # The reward is the lower, the higher the travelled distance is 
        if velocity > 0:
            reward -= 1e-2*self.traveled_distance
        else:
            reward -= 1e-1*self.traveled_distance
        
        # The reward is the lower, the higher the number of direction changes is
        direction_change = self.velocity_old*self.velocity_new < 0
        if direction_change:
            reward -= 1e3
        
        # Check if we jackknifed and induce a huge penalty for it
        jackknife = self.check_jackknife()
        if jackknife: 
            reward -= 1e3
            self.jackknive_number += 1
            
            if self.jackknive_number > 100:
                done = True
                reward -= 1e6
        
        # Checks if the trailer is out of bounds
        if self.check_out_of_bounds():
            reward -= 1e18
            done = True
        
        if translation_error < 0.5:
            # Christmas bonus
            reward += 1e9
            # The reward is the lower, the higher the rotation error of the trailer is
            if self.number_trailers == 0:
                reward -= 1e3*np.abs(truck_absolute_rotation)
            elif self.number_trailers == 2:
                reward -= 1e3*np.abs(second_trailer_absolute_rotation)
            else:
                reward -= 1e3*np.abs(first_trailer_absolute_rotation)
            done = True
            
        if self.check_timeout():
            reward -= 1e9
            done = True
            
        if done:
            print("done")
            
        return reward, done

    def reset(self):
        
        initial_truck_translation = np.array([15,2.5],dtype=np.float32)
        initial_truck_rotation = 0
        initial_first_trailer_rotation = 0
        initial_second_trailer_rotation = 0
        initial_velocity = 0
        initial_steering_percentage = 0
        
        self.traveled_distance = 0 
        
        distance_to_target = np.linalg.norm(initial_truck_translation-self.destination_translation)
        self.maximal_traveled_distance = distance_to_target*2
        
        self.state = np.array([initial_truck_translation[0],\
                               initial_truck_translation[1],\
                               initial_truck_rotation,\
                               initial_first_trailer_rotation,\
                               initial_second_trailer_rotation,\
                               initial_velocity,\
                               initial_steering_percentage])
        return self.state
    
    def reset_random(self): 
        
        random_initial_truck_translation = [np.random.uniform(14, 16),np.random.uniform(4, 8)]
        random_initial_truck_rotation = np.random.uniform(np.deg2rad(-180),np.deg2rad(180))
        random_initial_first_trailer_rotation = np.random.uniform(np.deg2rad(-90),np.deg2rad(90))
        random_initial_second_trailer_rotation = np.random.uniform(np.deg2rad(-90),np.deg2rad(90))
        initial_velocity = 0
        initial_steering_percentage = 0
        
        self.traveled_distance = 0 
        
        distance_to_target = np.linalg.norm(random_initial_truck_translation-self.destination_translation)
        self.maximal_traveled_distance = distance_to_target*2
        
        self.state = np.array([random_initial_truck_translation[0],\
                               random_initial_truck_translation[1],\
                               random_initial_truck_rotation,\
                               random_initial_first_trailer_rotation,\
                               random_initial_second_trailer_rotation,\
                               initial_velocity,\
                               initial_steering_percentage])
        return self.state
    
    def check_timeout(self):
        return self.traveled_distance > self.maximal_traveled_distance
    
    def get_absolute_translation_and_rotation_of_truck_and_trailer(self):
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        truck_translation = np.array([truck_translation_x,truck_translation_y])
        
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
        
        truck_absolute_translation = truck_translation
        truck_absolute_rotation = truck_rotation
        first_trailer_absolute_translation = truck_absolute_translation\
                                             +hitch_vector_truck\
                                             -hitch_vector_first_trailer_truck
        first_trailer_absolute_rotation = truck_absolute_rotation\
                                          +first_trailer_rotation
        second_trailer_absolute_translation = first_trailer_absolute_translation\
                                              +hitch_vector_first_trailer_second_trailer\
                                              -hitch_vector_second_trailer
        second_trailer_absolute_rotation = first_trailer_absolute_rotation\
                                           +second_trailer_rotation
                                           
        absolute_translation_and_rotation = [truck_absolute_translation,\
                                             truck_absolute_rotation,\
                                             first_trailer_absolute_translation,\
                                             first_trailer_absolute_rotation,\
                                             second_trailer_absolute_translation,\
                                             second_trailer_absolute_rotation]
        
        return absolute_translation_and_rotation
        
    def visualize(self):
        truck_translation_x,\
        truck_translation_y,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation,\
        velocity,\
        steering_percentage = self.state
        
        truck_translation = np.array([truck_translation_x,truck_translation_y])
        
        visualisation_element = [truck_translation,\
                                 truck_rotation,\
                                 first_trailer_rotation,\
                                 second_trailer_rotation,\
                                 steering_percentage,\
                                 self.destination_translation,\
                                 self.destination_rotation,\
                                 self.number_trailers]
        
        if np.mod(self.plotting_number,self.plotting_interval) == 0:
            self.plotting_number = 1
            self.visualize_combination.run(visualisation_element)
        else:
            self.plotting_number += 1

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

