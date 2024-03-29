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
        #Define limit of the variables in the observation space. 
        high = np.array([50, 50, np.finfo(np.float32).max, 1, 1, np.pi/2, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.action_space = spaces.Discrete(6)
        
        self.viewer = None
        
        self.name = "TrailerReversingDiscrete"
        self.masscar = 1000
        self.wheelbase = 2.5
        
        #Limit speed to this value (m/s)
        self.max_speed = 0.5
        
        #Mag T seems to be force magnitude for the discrete cotntrol of velocyt
        self.mag_T = 20000
        self.kv = 1000
       
        #self.dt = 0.02  # seconds between state updates 
        self.dt = 0.1
        
        #Size of angular increase each timestep. 
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
        self.traveledDistance = 0
        
        #This is recalculated in the reset function. 
        self.maxTraveledDistance = 10000
        
        #Specify when timeout occurs. If the agent hasnt solved the problem within
        #this many timesteps, break the episode. 
        self.maxTimesteps = 2000
        
        #Reset to the initial state. 
        self.reset()
        
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
            
        #Indicate a direction change, for use in the reward function. 
        dirChange = False     
        if action ==1 or action ==2 or action ==3: 
            dirChange = True
            
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
        
        self.traveledDistance += np.abs(v)*self.dt
        
        delta = np.clip(delta, -np.pi/3, np.pi/3     )
       
        self.state = np.array([x, y, v, cos_theta, sin_theta, delta, theta_t], dtype=np.float32)
        
        reward, done = self.calc_reward(dirChange)
        
        #TODO: Here we should probably return an info dict, with information about timeout etc
        # https://github.com/openai/gym/issues/1230
        return self.state.copy(), reward, done, {}
    
    
    def checkJackknife(self): 
        """ Returns true if the system has jackknifed. """
        state = self.state
        
        #TODO: Use namedtuple here instead for more typesafety
        x, y, v, cos_theta, sin_theta, delta, theta_t = state
      
        return abs(theta_t) > self.jack_knife_angle

    def calc_reward(self, dirChange):
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
        dist = np.sqrt(trailer_cog[0]**2 + trailer_cog[1]**2 )
        #dist = abs(x)   #The car distance as target. 
        
        #Try positive reward when closer to the orgin. 
        #reward = 1/(0.3 + dist)
        
        #Squared distance might make the agent overvalue states far from target?
        reward = -dist
        
        #Penalize direction changes. 
        if dirChange:
            reward -= 1
        
        #Let reward be trailer distance to y axis. 
        
        done = dist < 0.9 
        
        if(done): 
            reward = 4000; 
        
        
        #Quick and dirty debug! Timeout is not a terminal state, but a state with 
        #a discounted next reward. We just do not know how to implement a timeout when using 
        # baselines . Timeout should not be a terminal state! https://arxiv.org/abs/1712.00378
       
        #Abort when jackknifing, could work when reward is positive.
        #or self.checkJackknife()
        done = done or self.check_timeout() 
        
        return reward, done

    def reset(self):
        
        #print('Last time index', self.currentTimeIndex)
        #Keep track if number of updates the current episode
        self.currentTimeIndex = 0
        self.traveledDistance =0
        # Set your desired initial condition:
        #init_x = 8
        #init_y = 6
        #init_rot = -10*np.pi/180
        
        init_x = np.random.uniform(14, 16)
        init_y = np.random.uniform(4, 8)
        init_rot = np.random.uniform(-10*np.pi/180, 10*np.pi/180)
        
        #Set max traveled distance before timeout as init distance to target times some margin. 
        distToTarget = np.sqrt(init_x**2 + init_y**2)
        self.maxTraveledDistance = distToTarget*3
        
        #Calculate a reasonable timelimit too 
        timeToTarget = distToTarget/self.max_speed
        
        self.maxTimesteps = timeToTarget/self.dt*3
        #print('Timesteps lim ', self.maxTimesteps, "Dist", self.maxTraveledDistance)
        
        self.state = np.array([init_x, init_y, 0, np.cos(init_rot), np.sin(init_rot), 0, 0], dtype=np.float32)
        return self.state
    
    def reset_rand(self): 
        """ Reset and initialize to random position. TODO: not fully implemented. """
        
        self.reset()
        
        #TODO: Call reset, and add some randomness 
        
         # Set your desired initial condition:
        init_x = np.random.uniform(14, 16)
        init_y = np.random.uniform(4, 8)
        init_rot = np.random.uniform(-10*np.pi/180, 10*np.pi/180)
        
       
        self.state = np.array([init_x, init_y, 0, np.cos(init_rot), np.sin(init_rot), 0, 0], dtype=np.float32)
        return self.state
    
    def check_timeout(self):
        """ Return true if the agent definitely has diverged. We think this happens if 
        it has traveled a too large distance, wrt. to its initial pose in relation to the target
        or if too long time as elapsed."""
        timeOut = self.currentTimeIndex > self.maxTimesteps
        
        lost = self.traveledDistance > self.maxTraveledDistance
        
        return (lost or timeOut)
        
    
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

