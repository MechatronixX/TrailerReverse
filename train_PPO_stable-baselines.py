import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from gymEnvironments.gym_reverse_variable_trailer_number import Reverse_variable_trailer_number_environment

def callback(_locals, _globals):    
    return True 

def train():
    # multiprocess environment
    n_cpu = 4
    env = SubprocVecEnv([lambda: Reverse_variable_trailer_number_environment() for i in range(n_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=np.int(1e6), log_interval=10, callback = callback)
    model.save("ppo2_trailer")
    
if __name__ == "__main__":
    train()