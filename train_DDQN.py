# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:12:56 2019

@author: root
"""

# Import dependencies
import torch
import numpy as np
from numpy import array
import gym
from collections import namedtuple
import DDQN
from DDQN import DDQNhelpers

from DDQN.dqn_model import DoubleQLearningModel, ExperienceReplay
import DDQN.dqn_model
from DDQN.DDQNhelpers import *

from IPython.core.debugger import set_trace
from gymEnvironments.bicycleEnv import *