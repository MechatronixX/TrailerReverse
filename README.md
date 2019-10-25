# TrailerReverse
Use reinforcement learning to reverse a model of a towing vehicle with a connected trailer. Python is used, together with mainly PyTorch and Tensorflow.

A model of a vehicle and trailer is intregrated in an Open AI gym environment. Attempts have been made to make this system reverse well using 
PPO2 from Stable Baseslines as well as PPO from https://github.com/nikhilbarhate99/PPO-PyTorch. Also, a DDQN implementation mainly based on content from
a course template of https://github.com/JulianoLagana/deep-machine-learning.git has been tried. 

So far, the system does not behave well but this repository should be a good starting point for further experiments. Run the files named train_XXX.py or run__XX.py
to train/run the system using different RL approaches. 

# Dependencies 
We recommend starting out with a fresh Anaconda environment for this. In this you will probably at least need to install 
Open AI gym, Stable Baselines together with Tensorflow 1.14, Pytorch and Torch. 

We used Windows to run these files, but it should work on other platforms. 

Here is a useful guide for Open AI gym on Windows: 
https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30
