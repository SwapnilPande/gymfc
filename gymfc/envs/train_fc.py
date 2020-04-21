import numpy as np 
import configparser

import gym 
from gym import wrappers, logger, spaces
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from gymfc.envs.fc_env import FlightControlEnv
from gymfc.envs.gym_env import AttitudeFlightControlEnv
TRAINING_CONFIG = "./training_params.config"

# Gradually decrease the learning rate as the training progresses.
def lr_callback(frac):
    return 0.0001 * frac

def main():

    # Load training parameters
    cfg = configparser.ConfigParser()
    cfg.read(TRAINING_CONFIG)
    params = cfg["PARAMETERS"]

    n_steps = int(params["N_steps"])
    noptepochs = int(params["Noptepochs"])
    nminibatches = int(params["Nminibatches"])
    gamma = float(params["Gamma"])
    lam = float(params["Lam"])
    total_timesteps = int(params["Total_timesteps"])

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    # Create a callback to save model checkpoints
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs_1/ckpts/',
                                             name_prefix='rl_model_2')

    # Create a separate evaluation environment
    eval_env = gym.make('attitude-fc-v0')

    # Callback to evaluate the model during training
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_1/best_model',
                                log_path='./logs_1/results', eval_freq=100000)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create training environment
    env = gym.make('attitude-fc-v0')

    model = PPO2(MlpPolicy, 
                env,
                n_steps=n_steps,
                learning_rate=lr_callback,
                noptepochs=noptepochs,
                nminibatches=nminibatches,
                gamma=gamma,
                lam=lam,
                tensorboard_log='./logs_1/tensorboard/ppo2/')

    # Train and save the model
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("./logs_1/saved_models/ppo2_1")

    env.close()


if __name__ == '__main__':
    main()
   
