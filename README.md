# Project: Navigation from Udacity Deep Reinforcement Learning

- [Project: Navigation from Udacity Deep Reinforcement Learning](#project-navigation-from-udacity-deep-reinforcement-learning)
  - [Introduction](#introduction)
  - [Description of the project](#description-of-the-project)
  - [Getting started and setting up the environment](#getting-started-and-setting-up-the-environment)
  - [Instructions](#instructions)


## Introduction

In this project we will train a deep reinforcement learning agent that is able to navigate autonomously in a simple world and knows how to collect bananas.
The described environment is modeled by Unity and modified by Udacity.
We will use a DQ-Network similar to that described in the paper [Human level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

![Banana World](banana_world.gif)

## Description of the project

We will write a DQN algorithm to train a reinforcement learning agent and optimise its hyperparameters. The agent shall be able to collect only the yellow bananas as they give +1 reward while avoiding the blue ones (-1 reward). The ray-based perceptions of objects in front of the agent are provided as continuous states with 37 dimensions (floats) and also contains the agents velocity.

Only given the information of the states, the agent has to figure out how to act by itself. 
It can choose one of four discrete actions:
* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

So before training it doesn't know how the enviroment reacts (type of reward or punishment in what circumstances). After many episodes it will gain more and more experience. In this project we consider an average score of +13 over 100 consecutive episodes as enough experience.

## Getting started and setting up the environment

1. Clone or download the Udacity [DLRLND repository](https://github.com/udacity/deep-reinforcement-learning) 
2. Please follow the instructions in the DRLND GitHub repository to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch and a few more Python packages required to run this repository. You don't need to install the ML Agent's toolkit. 
3. Download a prepared Unity environment (modified by Udacity) for your operating system: 
   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - [Linux headless](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
   - [Windows](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
   - [MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
4. Unzip the file into the p1_navigation/ folder.
5. Download from this repository and place into the p1_navigation/ folder:
   1. the Navigation.ipynb jupyter notebook containing all the code
   2. and checkpoint.pth of containing the trained model weights of the agent.
6. Adjust the path to the banana environment in the notebook file.

Hints for usage with WSL2: 
- it is recommended to use the Linux headless version as it can be alot of work to setup your visualisation dependencies.
- Use chmod -R 755 to grant permission to the Banana environment setup file (eg. Banana.x86_64)

## Instructions
1. Open the Navigation.ipynb as a jupyter notebook file
2. Use options below running the according cells in the notebook file.

| Option                               | Code cell |
| ------------------------------------ | ---  |
| Load the environment                 | 1-3  |
| Change hyperparameters               | 4    |
| Train the agent                      | 5    |  
| Save the model weights               | 6    |
| Let the agent play                   | 7-8  | 
| Close the environment                | 9    |