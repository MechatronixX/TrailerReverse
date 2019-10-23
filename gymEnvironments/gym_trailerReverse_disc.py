import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
#from utility_functions import out_of_bounds

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
        
        self.name = "TrailerReversingDiscrete"
        self.masscar = 1000
        self.wheelbase = 2.5
        
        #Limit speed to this value (m/s)
        self.max_speed = 1
        
        #Mag T seems to be force magnitude for the discrete cotntrol of velocyt
        self.mag_T = 20000
        self.kv = 1000
        self.dt = 0.02  # seconds between state updates
        self.ddelta_mag = 3*np.pi/2*self.dt
        self.trailer_len = 2
        
        #This defines the square box in the rendering? 
        self.target_position = np.array([self.trailer_len/2 + 2, 6])
        self.bar_len = 1
        self.trailer_bar_combo_len = self.trailer_len + self.bar_len
        self.jack_knife_angle = np.pi/2
        self.world_width = 20
        self.world_heigth = 12
        
    def step(self, action):
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
        T = 0
        ddelta = 0
        if action==1: # action=1 <=> backward
            T = -self.mag_T
        elif action == 2:
            ddelta = self.ddelta_mag
        elif action == 3:
            ddelta = -self.ddelta_mag
            
        v = v + self.dt/self.masscar * (T - self.kv*v)
        
        if abs(v) < 0.03:
            v = 0
            
        #If we jackknife, imagine a squeeking sound and the car has to stop. This implies
        #more long term penaly as we have to accelerate again now. 
        #if (self.checkJackknife() ): 
        #    v =0 
            
        #Clamp velocity within reasonanle boundaries.     
        v = np.clip(v, -self.max_speed, self.max_speed)    
            
        # Makes sure steering angle doesn't get too big:
        if abs(delta + ddelta) < np.pi/3:
            delta = delta + ddelta

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
        state = self.state
        x, y, v, cos_theta, sin_theta, delta, theta_t = state
        
        car_cog, car_abs_rot, trailer_cog, trailer_abs_rot = self.get_absolute_orientation_and_cog_of_truck_and_trailer()
        
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
        dist = abs(x)   #The car distance as target. 
        reward = -dist
        
        #Let reward be trailer distance to x axis. 
        
        done = dist < 0.1 
        
        return reward, done

    def reset(self):
        # Set your desired initial condition:
        init_x = 15
        init_y = 6
        init_rot = -10*np.pi/180
        
        self.state = np.array([init_x, init_y, 0, np.cos(init_rot), np.sin(init_rot), 0, 0], dtype=np.float32)
        return self.state
    
    def reset_rand(self): 
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
        
    def render(self):
        screen_width = 800
        screen_height = int(screen_width * self.world_heigth/self.world_width)

        scale = screen_width/self.world_width
        carwidth = self.wheelbase * scale
        carheight = 0.64 * self.wheelbase * scale
        tirewidth = 0.8 * scale
        tireheight = 0.2 * scale
        target_position_scaled = self.target_position * scale
        trailer_bar_width = self.bar_len * scale
        trailer_bar_height = trailer_bar_width/10
        trailer_width = self.trailer_len * scale
        trailer_height = carheight
        
        

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Target:
            l,r,t,b = -trailer_width/2, trailer_width/2, trailer_height/2, -trailer_height/2
            target = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            target.set_color(0.8,1,0.8)
            self.target_trans = rendering.Transform(translation=(target_position_scaled[0], target_position_scaled[1]))
            target.add_attr(self.target_trans)
            self.viewer.add_geom(target)
            # Car
            l,r,t,b = -carwidth/2, carwidth/2, carheight/2, -carheight/2
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.set_color(1,0,0)
            self.car_trans = rendering.Transform()
            car.add_attr(self.car_trans)
            self.viewer.add_geom(car)
            # Tire left front (lf):
            l,r,t,b = -tirewidth/2, tirewidth/2, tireheight/2, -tireheight/2
            tirelf = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tirelf.set_color(0,0,0)
            self.tirelf_trans = rendering.Transform(translation=(carwidth/2, carheight/2))
            tirelf.add_attr(self.tirelf_trans)
            tirelf.add_attr(self.car_trans)
            self.viewer.add_geom(tirelf)
            # Tire right front (rf):
            l,r,t,b = -tirewidth/2, tirewidth/2, tireheight/2, -tireheight/2
            tirerf = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tirerf.set_color(0,0,0)
            self.tirerf_trans = rendering.Transform(translation=(carwidth/2, -carheight/2))
            tirerf.add_attr(self.tirerf_trans)
            tirerf.add_attr(self.car_trans)
            self.viewer.add_geom(tirerf)
            # Tire left rear (lr):
            l,r,t,b = -tirewidth/2, tirewidth/2, tireheight/2, -tireheight/2
            tirelr = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tirelr.set_color(0,0,0)
            self.tirelr_trans = rendering.Transform(translation=(-carwidth/2, carheight/2))
            tirelr.add_attr(self.tirelr_trans)
            tirelr.add_attr(self.car_trans)
            self.viewer.add_geom(tirelr)
            # Tire right rear (rr):
            l,r,t,b = -tirewidth/2, tirewidth/2, tireheight/2, -tireheight/2
            tirerr = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tirerr.set_color(0,0,0)
            self.tirerr_trans = rendering.Transform(translation=(-carwidth/2, -carheight/2))
            tirerr.add_attr(self.tirerr_trans)
            tirerr.add_attr(self.car_trans)
            self.viewer.add_geom(tirerr)
            # Trailer bar:
            l,r,t,b = -trailer_bar_width/2, trailer_bar_width/2, trailer_bar_height/2, -trailer_bar_height/2
            trailer_bar = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            trailer_bar.set_color(0,0,0)
            self.trailer_bar_trans = rendering.Transform()
            trailer_bar.add_attr(self.trailer_bar_trans)
            self.viewer.add_geom(trailer_bar)
            # Trailer:
            l,r,t,b = -trailer_width/2, trailer_width/2, trailer_height/2, -trailer_height/2
            trailer = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            trailer.set_color(0.5,0.5,0.5)
            self.trailer_trans = rendering.Transform()
            trailer.add_attr(self.trailer_trans)
            self.viewer.add_geom(trailer)
            # Trailer wheel left:
            l,r,t,b = -tirewidth/2, tirewidth/2, tireheight/2, -tireheight/2
            tire_t_l = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tire_t_l.set_color(0,0,0)
            self.tire_t_l_trans = rendering.Transform(translation=(0, trailer_height/2))
            tire_t_l.add_attr(self.tire_t_l_trans)
            tire_t_l.add_attr(self.trailer_trans)
            self.viewer.add_geom(tire_t_l)
            # Trailer wheel right:
            l,r,t,b = -tirewidth/2, tirewidth/2, tireheight/2, -tireheight/2
            tire_t_r = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tire_t_r.set_color(0,0,0)
            self.tire_t_r_trans = rendering.Transform(translation=(0, -trailer_height/2))
            tire_t_r.add_attr(self.tire_t_r_trans)
            tire_t_r.add_attr(self.trailer_trans)
            self.viewer.add_geom(tire_t_r)
            
            


        if self.state is None: return None

        x, y, v, cos_theta, sin_theta, delta, theta_t = self.state
        car_cog, car_abs_rot, trailer_cog, trailer_abs_rot = self.get_absolute_orientation_and_cog_of_truck_and_trailer()
        x_cog, y_cog = car_cog
        rear_center = np.array([x, y], dtype=np.float32)
        # Manual conversion from rear-wheel axis CoG of car:
        self.car_trans.set_translation(x_cog*scale, y_cog*scale)
        self.car_trans.set_rotation(np.arctan2(sin_theta, cos_theta))
        # tire rotations:
        self.tirelf_trans.set_rotation(delta)
        self.tirerf_trans.set_rotation(delta)
        # trailer movement (sort of messy, I could have implemented relative
        # translations but this went quicker to implement for me at least)
        trailer_bar_cog = rear_center - self.bar_len/2*np.array([np.cos(trailer_abs_rot), np.sin(trailer_abs_rot)], dtype=np.float32)
        self.trailer_bar_trans.set_translation(trailer_bar_cog[0]*scale, trailer_bar_cog[1]*scale)
        self.trailer_bar_trans.set_rotation(trailer_abs_rot)
        self.trailer_trans.set_translation(trailer_cog[0]*scale, trailer_cog[1]*scale)
        self.trailer_trans.set_rotation(trailer_abs_rot)
        
        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

