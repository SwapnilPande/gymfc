### ----------------------------------------------------------------------------- ###
#
# This script will load a trained flight controller model and evaluate its result. 
# Main purpose is to be viewed in Gazebo.
# 
# To view the model in Gazebo while this script is running in a Docker container, install
# the same version of Gazebo on your local computer. After executing this script in Docker,
# note the GAZEBO_MASTER_URI <PORT> printed to console. Then enter this locally:
# 
# export GAZEBO_IP=172.17.0.1 GAZEBO_MASTER_URI=172.17.0.2:<PORT>  
# gzclient --verbose
#
# Script runs until Ctrl-C is entered.
#
### ----------------------------------------------------------------------------- ###

import numpy as np 
import configparser

import gym 
from gym import wrappers, spaces
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from gymfc.envs.fc_env import FlightControlEnv
from gymfc.envs.gym_env import AttitudeFlightControlEnv


def main():

    # Create an evaluation environment
    env = gym.make('attitude-fc-v0')

    # Load the model
    model = PPO2.load("./logs/model_level_trained/best_model/best_model")

    # Evaluate the agent
    obs = env.reset()
    while True:
        try:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        except KeyboardInterrupt:
            print("INFO: Ctrl-C caught. Cleaning up...")
            env.close()
        

if __name__ == '__main__':
    main()
   
