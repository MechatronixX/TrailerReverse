import numpy as np
import time
import msvcrt
#from simple_template import CartParkingEnv
#env = CartParkingEnv()
from trailer_reverse_gym_environment_no_reward import CarTrailerParkingRevEnv
env = CarTrailerParkingRevEnv()

env.reset()
rev_sum = 0
for i in range(2500):
    action = 0
    if msvcrt.kbhit():
        input_char = msvcrt.getch()
        if input_char == "s".encode():
            action = 1
        elif input_char == "a".encode():
            action = 2
        elif input_char == "d".encode():
            action = 3
    
    state, reward, done, _ = env.step(action)
    rev_sum = rev_sum + reward
    print('Current reward:', reward, '\t Sum:', rev_sum)
    if not done:
        env.render()
    else:
        break
env.close()