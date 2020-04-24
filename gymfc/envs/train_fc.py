### -------------------------------------------------------------------------- ###
# 
#  This script trains a quadcopter flight controller using the PPO2 stable baselines
#  algorithm. It uses the AttitudeFlightControllerEnv gym environment that was developed to 
#  interface with the GymFC base environment.
# 
#  PPO2 stable baselines documentation: https://www.google.com/search?q=ppo2+stable+baselines&oq=ppo2+stable+baselines&aqs=chrome..69i57j69i60l3.2423j0j7&sourceid=chrome&ie=UTF-8
#
### -------------------------------------------------------------------------- ###

import numpy as np 
import configparser
import datetime
import shutil
import os

# RL packages
import gym 
from gym import wrappers, logger, spaces
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from gymfc.envs.fc_env import FlightControlEnv
from gymfc.envs.gym_env import AttitudeFlightControlEnv

# Training hyperparameters config file
TRAINING_CONFIG = "./training_params.config"

# Gradually decrease the learning rate as the training progresses.
def create_lr_callback(lr_max, lr_min):
    def lr_callback(frac):
        return (lr_max - lr_min) * frac + lr_min 
    return lr_callback

# Create a unique directory to save model training data.
def create_model_log_dir():
    date = datetime.datetime.now()

    model_log_dir = "model_" + str(date.year) + "_" + str(date.month) + "_" \
        + str(date.day) + "_" + str(date.hour) + "_" + str(date.minute)

    os.mkdir("./logs/" + model_log_dir)
    return model_log_dir

def main():

    # Load training parameters
    cfg = configparser.ConfigParser()
    cfg.read(TRAINING_CONFIG)
    params = cfg["PARAMETERS"]

    learning_rate_max = float(params["learning_rate_max"])
    learning_rate_min = float(params["learning_rate_min"])
    n_steps = int(params["N_steps"])
    noptepochs = int(params["Noptepochs"])
    nminibatches = int(params["Nminibatches"])
    gamma = float(params["Gamma"])
    lam = float(params["Lam"])
    total_timesteps = int(params["Total_timesteps"])

    lr_callback = create_lr_callback(learning_rate_max, learning_rate_min)

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    # Create save directory and various save paths
    model_log_dir = create_model_log_dir()
    save_path = "./logs/" + model_log_dir + "/ckpts/"
    best_model_save_path = "./logs/" + model_log_dir + "/best_model/"
    log_path = "./logs/" + model_log_dir + "/results/"
    tensorboard_dir = "./logs/" + model_log_dir + "/tensorboard/"
    model_save_path = "./logs/saved_models/" + model_log_dir

    # Save training and reward params to model directory 
    shutil.copy("./gymfc/reward_params.config", "./logs/" + model_log_dir + "/reward_params.config")
    shutil.copy("./training_params.config", "./logs/" + model_log_dir + "/training_params.config")

    # Create a callback to save model checkpoints
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path,
                                             name_prefix='rl_model')

    # Create a separate evaluation environment
    eval_env = gym.make('attitude-fc-v0')

    # Callback to evaluate the model during training
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                log_path=log_path, eval_freq=100000)

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create training environment
    env = gym.make('attitude-fc-v0')

    # RL Agent
    model = PPO2(MlpPolicy, 
                env,
                n_steps=n_steps,
                learning_rate=lr_callback,
                noptepochs=noptepochs,
                nminibatches=nminibatches,
                gamma=gamma,
                lam=lam,
                cliprange=0.1,
                tensorboard_log=tensorboard_dir)

    # Train and save the model
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_save_path)

    env.close()


if __name__ == '__main__':
    main()
   
