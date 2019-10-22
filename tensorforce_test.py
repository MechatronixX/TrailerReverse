# -*- coding: utf-8 -*-

from tensorforce import *
from tensorforce.agents import Agent

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    states=dict(type='float', shape=(10,)),
    actions=dict(type='int', num_values=5),
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

# Initialize the agent
agent.initialize()

# Retrieve the latest (observable) environment state
state = get_current_state()  # (float array of shape [10])

# Query the agent for its action decision
action = agent.act(states=state)  # (scalar between 0 and 4)

# Execute the decision and retrieve the current performance score
reward = execute_decision(action)  # (any scalar float)

# Pass feedback about performance (and termination) to the agent
agent.observe(reward=reward, terminal=False)