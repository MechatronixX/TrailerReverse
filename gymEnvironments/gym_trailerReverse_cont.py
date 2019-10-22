import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from utility_functions import out_of_bounds

from visualisation.Visualize_combination_code import *
from simulation.Simulate_combination_code import *
from helperfunctions.load_shapes_code import *

class CarTrailerParkingRevEnv(gym.Env):
    """
    Description:

    Observation: 
        
    Actions:
        Type: Discrete(4)

    Reward:


    Starting State:


    Episode Termination:

    """
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 50}

    def __init__(self):
        high = np.array([50, 50, np.finfo(np.float32).max, 1, 1, np.pi/2, np.finfo(np.float32).max])
        self.state = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.viewer = None
        
        self.masscar = 1000
        self.wheelbase = 2.5
        self.mag_T = 20000
        self.kv = 1000
        self.dt = 0.02  # seconds between state updates
        self.ddelta_mag = 3*np.pi/2*self.dt
        self.trailer_len = 2
        self.target_position = np.array([self.trailer_len/2 + 2, 6])
        self.bar_len = 1
        self.trailer_bar_combo_len = self.trailer_len + self.bar_len
        self.jack_knife_angle = np.pi/2
        self.world_width = 20
        self.world_heigth = 12
        
        self.number_trailers = 1
        
        self.visualisation_shapes = load_shapes(self.number_trailers)

        self.visualize_combination = Visualize_combination(self.visualisation_shapes)
        
        # Do not change
        self.plotting_number = 1
        
        # Plotting interval in steps
        self.plotting_interval = 1e2
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        truck_translation_x,\
        truck_translation_y,\
        velocity,\
        truck_rotation_cos,\
        truck_rotation_sin,\
        delta,\
        first_trailer_rotation = self.state
        
        second_trailer_rotation = 0
        step_size = self.dt
        
        truck_translation = [truck_translation_x,\
                             truck_translation_y]
        
        truck_rotation = np.rad2deg(np.arccos(truck_rotation_cos))
        
        simulate_combination = Simulate_combination(visualisation_shapes,\
                                                    truck_translation,\
                                                    truck_rotation,\
                                                    first_trailer_rotation,\
                                                    second_trailer_rotation,\
                                                    self.number_trailers,\
                                                    step_size)
        

        
        #This seems to be the steering angle increment. 
        T = 0
        ddelta = 0
        if action==1: # action=1 <=> backward
            T = -self.mag_T
        elif action == 2:
            ddelta = self.ddelta_mag
        elif action == 3:
            ddelta = -self.ddelta_mag
            
        velocity = velocity + self.dt/self.masscar * (T - self.kv*velocity)
        if abs(velocity) < 0.03:
            velocity = 0
        # Makes sure steering angle doesn't get too big:
        if abs(delta + ddelta) < np.pi/3:
            delta = delta + ddelta
            
        steering_percentage = np.rad2deg(delta)/60
        
        truck_translation,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation = self.simulate_combination.run(velocity,steering_percentage)
        
        truck_translation_x,\
        truck_translation_y = truck_translation
        
        truck_rotation_cos = np.cos(np.deg2rad(truck_rotation))
        truck_rotation_sin = np.sin(np.deg2rad(truck_rotation))
        
        self.state = np.array([truck_translation_x,\
                               truck_translation_y,\
                               velocity,\
                               truck_rotation_cos,\
                               truck_rotation_sin,\
                               delta,\
                               first_trailer_rotation], dtype=np.float32)
    
        reward, done = self.calc_reward()
        
        self.state = np.array([2, 2, 1], dtype=np.float32)
        return self.state.copy(), reward, done, {}
    
    def calc_reward(self):
        done = False
        state = self.state
        x, y, v, cos_theta, sin_theta, delta, theta_t = state
        car_cog, car_abs_rot, trailer_cog, trailer_abs_rot = self.get_absolute_orientation_and_cog_of_truck_and_trailer()
        
        #if abs(theta_t) >= self.jack_knife_angle:
        #    done = True
        #if out_of_bounds([car_cog, trailer_cog], [180/np.pi*car_abs_rot, 180/np.pi*trailer_abs_rot],
        #                 yard_shape = np.array([self.world_width, self.world_heigth])):
        #    done = True
        
        
        
        dist = np.abs(trailer_cog[0])
        #Let reward be trailer distance to x axis. 
        reward = -(trailer_cog[0]**2)
        
        done = dist < 0.1 
        
        return reward, done

    def reset(self):
        # Set your desired initial condition:
        init_x = np.random.uniform(14, 16)
        init_y = np.random.uniform(4, 8)
        init_rot = np.random.uniform(-10*np.pi/180, 10*np.pi/180)
        
        self.state = np.array([init_x, init_y, 0, np.cos(init_rot), np.sin(init_rot), 0, 0], dtype=np.float32)
        return self.state
    
    def get_absolute_orientation_and_cog_of_truck_and_trailer(self):
        x, y, v, cos_theta, sin_theta, delta, theta_t = self.state
        rear_center = np.array([x, y], dtype=np.float32)
        car_cog = rear_center + self.wheelbase/2 * np.array([cos_theta, sin_theta], dtype=np.float32)
        car_abs_rot = np.arctan2(sin_theta, cos_theta)
        trailer_abs_rot = car_abs_rot + theta_t
        trailer_cog = rear_center - (self.bar_len + self.trailer_len/2)*np.array([np.cos(trailer_abs_rot), np.sin(trailer_abs_rot)], dtype=np.float32)
        
        return car_cog, car_abs_rot, trailer_cog, trailer_abs_rot
        
    def render(self,visualisation_element):
            
        if np.mod(self.plotting_number,self.plotting_interval) == 0:
            self.plotting_number = 1
            self.visualize_combination.run(visualisation_element)
        else:
            self.plotting_number += 1
        

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

