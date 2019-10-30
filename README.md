# TrailerReverse
Use reinforcement learning to reverse a model of a towing vehicle with a connected trailer. Python is used, with PyTorch and Tensorflow for machine learning capabilities.

A model of a vehicle and trailer is intregrated in an Open AI gym compatible environment. Attempts have been made to make this system reverse well using 
PPO2 from Stable Baseslines as well as PPO from https://github.com/nikhilbarhate99/PPO-PyTorch. Also, a DDQN implementation mainly based on content from
a course template of https://github.com/JulianoLagana/deep-machine-learning.git has been tried. 

# Dependencies 
As seems to be common within machine learning as of today, it can take some work to install the correct dependencies and get all to run properly. Python has a thriving open source community, where many independent contributions might not be compatible with one another, or your operative system. 

We recommend starting out with a fresh Anaconda environment for this project. In this you will probably at least need to install 
Open AI gym, Stable Baselines together with Tensorflow 1.14, Pytorch and Torch. With the proper Anaconda environment activated, hopefully these installs within your Anaconda prompt should be enough to get started:

```
pip install gym 
conda install torch
pip install pytorch
pip install tensorflow==1.14
pip install stable-baselines 
```

We used Windows to run these files, but it should work on other platforms. 

Here is a useful guide for Open AI gym on Windows: 
https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30

# Getting started
There are two custom AI gym environments created that should possible to use in training loops designed for other AI gym environments. 

A good start could be to run the [gymEnvironments/keyboardSimulation.py](gymEnvironments/keyboardSimulation.py). Run this in your system terminal, and not inline in for instance Spyder, else the keyreads probably wont work. Analyze the code in [gymEnvironments/keyboardSimulation.py](gymEnvironments/keyboardSimulation.py) to figure out which keyboard press does what. You should see a pop-up as such and see a car and trailer move around:    

![Text](doc/figures/carAndTrailer.PNG?raw=true  "Car and trailer environment")

## Training 
Run the files named train_XXX.py to train a network on the the environment. 
So far, the system does not behave well but this repository should be a good starting point for further experiments. Run the files named  or run__XX.py
to train/run the system using different RL approaches. 

# Environments 
There are two local open AI gym environments included in the repository. One is a car with a small trailer connnecter, and the other is a truck with a dolly and long trailer. These models are implemented using somewhat different approaches, in case you choose to modify these models and get confused. 

# TODO
* Figure out what is a reasonable cost function for this 
* Try different RL algorithms with different hyper parameters 
* Ensure that timeout is handled properly https://github.com/openai/gym/issues/1230
