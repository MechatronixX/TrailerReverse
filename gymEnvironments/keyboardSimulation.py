import numpy as np
import time
import msvcrt
import time
#from simple_template import CartParkingEnv
#env = CartParkingEnv()
#from gymEnvironments.gym_trailerReverse_disc import CarTrailerParkingRevEnv

"""
Keyboard simulation of the environment to get an idea of how it behaves. Run in conda terminal or so 
and not in Spyder, as spyder has problems catching the keystrokes in real time. 
"""
from gym_trailerReverse_disc import CarTrailerParkingRevEnv
env = CarTrailerParkingRevEnv()

env.reset()
rev_sum = 0

FPS = 2 
Ts = 1/FPS

for i in range(2500):
    action = 0 #Deafult action, do nothing. 
    if msvcrt.kbhit():
        input_char = msvcrt.getch()
        if input_char == "q".encode():
            action = 1 #Stopa
        elif input_char == "w".encode():
            action = 2 #Forward
        elif input_char == "s".encode():
            action = 3 #Reverse
        elif input_char == "a".encode():
            action = 4 #inc steering
        elif input_char == "d".encode():
            action = 5 #dec steering
    
    state, reward, done, _ = env.step(action)
    rev_sum = rev_sum + reward
    print('Current reward:', reward, '\t Sum:', rev_sum)
    
    #time.sleep(Ts)
    
    if not done:
        env.render()
    else:
        print("Reached terminal state")
        break
env.close()