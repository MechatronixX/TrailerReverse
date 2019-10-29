import gym
import numpy as np
from scipy.integrate import odeint

from visualisation.Visualize_combination_code import *
from helperfunctions.constant_rotation_code import *
from helperfunctions.load_shapes_code import *

class CarTrailerParkingRevEnv(gym.Env):
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 25}

    def __init__(self):
        high = np.array([50, 50, np.finfo(np.float32).max, 1, 1, np.pi/2, np.finfo(np.float32).max])
        self.state = None
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.viewer = None
        
        self.name = "TrailerReversingDiscrete"
        self.masscar = 1000
        self.wheelbase = 2.5
        
        #Limit speed to this value (m/s)
        self.max_speed = 1
        
        #Mag T seems to be force magnitude for the discrete cotntrol of velocyt
        self.mag_T = 20000
        self.kv = 1000
        #self.dt = 0.02  # seconds between state updates
        
        self.dt = 0.08
        
        #Size of angular increase each timestep. 
        #self.ddelta_mag = 3*np.pi/2*self.dt
        self.ddelta_mag = np.deg2rad(1)
        self.trailer_len = 2
        
        #This defines the square box in the rendering? 
        self.target_position = np.array([self.trailer_len/2 + 2, 6])
        self.bar_len = 1
        self.trailer_bar_combo_len = self.trailer_len + self.bar_len
        self.jack_knife_angle = np.pi/2
        self.world_width = 20
        self.world_heigth = 12
        
        self.currentTimeIndex = 0
        
        #Watchdog, so the vehicle doesnt wander around for too long, missing the target. 
        self.traveled_distance = 0
        self.maximal_traveled_distance = 10000
        
        #Specify when timeout occurs. If the agent hasnt solved the problem within
        #this many timesteps, break the episode. 
        self.timeOut = 2000
        
        #Reset to the initial state. 
        self.reset()
        
        #######################################################################
        
        number_trailers = 2
        visualisation_shapes = load_shapes(number_trailers)
        self.visualize_combination = Visualize_combination(visualisation_shapes)
        
    def step(self, action):
        
        self.currentTimeIndex+=1
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        #Sine/cos of the angle are in the state vector, hmm. 
        x, y, v, cos_theta, sin_theta, delta, theta_t = self.state
        dtheta = v*np.tan(delta)/self.wheelbase
        
        ## Integration of the differential equations. This seems to be a vector with the derivatives. 
        def s_cont_dot(s_cont, t):
            return np.array([v*s_cont[2], 
                             v*s_cont[3], 
                             -dtheta*s_cont[3], 
                             dtheta*s_cont[2],
                             -dtheta + v/self.trailer_bar_combo_len*np.sin(-s_cont[4])])
        
        
        s_cont = np.array([x, y, cos_theta, sin_theta, theta_t], dtype=np.float32)
        t_array = np.array([0, self.dt])
        s_cont = odeint(s_cont_dot, s_cont, t_array)[1] # integration
        x = s_cont[0]
        y = s_cont[1]
        cos_theta = s_cont[2]
        sin_theta = s_cont[3]
        theta_t = s_cont[4]
        # To ensure that cos_theta^2 + sin_theta^2 = 1. (I tested without it and it doesn't diverge, but still it doesn't hurt)
        theta = np.arctan2(sin_theta, cos_theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        #It seems there are for discrete actions. Two of them changes the vehicle velocity, 
        #and two of them changes the steering angle. 
        #This sseems to be the steering angle increment. 
        F = 0
        ddelta = 0
        if action ==0: 
            None 
            #Do nothing. 
        elif action==1: 
            v = 0 #Brake
        elif action == 2:
            v = self.max_speed #Forward
            F = 1
        elif action == 3:
            v =-self.max_speed #Reverse
            F = -1
        elif action == 4: 
            delta += self.ddelta_mag
        elif action == 5: 
            delta -= self.ddelta_mag
            
        #v = v + self.dt/self.masscar * (T - self.kv*v)
        
        #if abs(v) < 0.03:
        #    v = 0
            
        #If we jackknife, imagine a squeeking sound and the car has to stop. This implies
        #more long term penaly as we have to accelerate again now. If force T is larger than 0
        #we are trying to drive foward to avoid the jacknife, so dont zero velocity out then.
        if (self.checkJackknife() and F <= 0 ): 
            v =0 
            
        #Clamp velocity within reasonanle boundaries.     
        v = np.clip(v, -self.max_speed, self.max_speed)  
        
        self.traveled_distance += v*self.dt
        
        delta = np.clip(delta, -np.pi/3, np.pi/3     )
       
        self.state = np.array([x, y, v, cos_theta, sin_theta, delta, theta_t], dtype=np.float32)
        
        reward, done = self.calc_reward()
        
        return self.state.copy(), reward, done, {}
    
    
    def checkJackknife(self): 
        """ Returns true if the system has jackknifed. """
        state = self.state
        x, y, v, cos_theta, sin_theta, delta, theta_t = state
      
        return abs(theta_t) > self.jack_knife_angle

    def calc_reward(self):
        done = False
        x, y, v, cos_theta, sin_theta, delta, theta_t = self.state
        
        car_cog, car_abs_rot, trailer_cog, trailer_abs_rot = self.get_absolute_translation_and_rotation_of_truck_and_trailer()
        
        #baseReward = -(trailer_cog[0]**2)
        #baseReard = -(np.abs(trailer_cog[0]))
        
        #Check if we jackknifed and induce a huge penalty for it. 
        #jackknife = self.checkJackknife()
        
        #if jackknife: 
        #    reward = baseReward*100
        #else:
        #reward = baseReward
        
        
        #if abs(theta_t) >= self.jack_knife_angle:
        #    done = True
        #if out_of_bounds([car_cog, trailer_cog], [180/np.pi*car_abs_rot, 180/np.pi*trailer_abs_rot],
        #                 yard_shape = np.array([self.world_width, self.world_heigth])):
        #    done = True
        
        
        #TODO: Should rather be the wheel axis center?? 
        #dist = np.abs(trailer_cog[0])
        dist = np.sqrt(trailer_cog[0]**2 + trailer_cog[1]**2 )
        #dist = abs(x)   #The car distance as target. 
        
        #Try positive reward when closer to the orgin. 
        reward = 1/(0.3 + dist)
        
        #reward = -dist*dist
        
        #Let reward be trailer distance to y axis. 
        
        done = dist < 0.3 
        
        #Quick and dirty debug! Timeout is not a terminal state, but a state with 
        #a discounted next reward. We just do not know how to implement a timeout when using 
        # baselines . Timeout should not be a terminal state!
        timeOut = self.currentTimeIndex > self.timeOut
        
        #Abort when jackknifing, could work when reward is positive. 
        done = done or self.check_timeout() or self.checkJackknife()
        
        return reward, done

    def reset(self):
        
        self.currentTimeIndex = 0
        
        initial_truck_translation = array([20,3.5],dtype=np.float32)
        initial_truck_rotation = 0
        initial_first_trailer_rotation = 0
        initial_second_trailer_rotation = 0
        
        self.traveled_distance = 0 
        
        distance_to_target = np.norm(initial_truck_translation-self.destination_translation)
        self.maximal_traveled_distance = distance_to_target*2
        
        self.state = np.array([initial_truck_translation,initial_truck_rotation,initial_first_trailer_rotation,initial_second_trailer_rotation],dtype=np.float32)
        return self.state
    
    def reset_random(self): 
        
        self.currentTimeIndex = 0
        
        random_initial_truck_translation = [np.random.uniform(14, 16),np.random.uniform(4, 8)]
        random_initial_truck_rotation = np.random.uniform(np.deg2rad(-180),np.deg2rad(180))
        random_initial_first_trailer_rotation = np.random.uniform(np.deg2rad(-90),np.deg2rad(90))
        random_initial_second_trailer_rotation = np.random.uniform(np.deg2rad(-90),np.deg2rad(90))
        
        self.traveled_distance = 0 
        
        distance_to_target = np.norm(random_initial_truck_translation-self.destination_translation)
        self.maximal_traveled_distance = distance_to_target*2
        
        self.state = np.array([random_initial_truck_translation,random_initial_truck_rotation,random_initial_first_trailer_rotation,random_initial_second_trailer_rotation],dtype=np.float32)
        return self.state
    
    def check_timeout(self):
        return self.traveled_distance > self.maximal_traveled_distance
    
    def get_absolute_translation_and_rotation_of_truck_and_trailer(self):
        truck_translation,\
        truck_rotation,\
        first_trailer_rotation,\
        second_trailer_rotation = self.state
        
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
                                             +hitch_vector_first_trailer_truck
        first_trailer_absolute_rotation = truck_absolute_rotation\
                                          +first_trailer_rotation
        second_trailer_absolute_translation = first_trailer_absolute_translation\
                                              +hitch_vector_first_trailer_second_trailer\
                                              +hitch_vector_second_trailer
        second_trailer_absolute_rotation = first_trailer_absolute_rotation\
                                           +second_trailer_rotation
        
        return truck_absolute_translation,truck_absolute_rotation,first_trailer_absolute_translation,first_trailer_absolute_rotation,second_trailer_absolute_translation,second_trailer_absolute_rotation
        
    def visualize(self):
        visualisation_element = [self.truck_translation,\
                                 self.truck_rotation,\
                                 self.first_trailer_rotation,\
                                 self.second_trailer_rotation,\
                                 self.steering_percentage,\
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

