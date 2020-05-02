### -------------------------------------------------------------------------- ###
# 
#  This script trains a quadcopter flight controller using stable baselines
#  algorithms. It uses the AttitudeFlightControllerEnv gym environment that was developed to 
#  interface with the GymFC base environment.
#
#  Run this script from the root directory.
# 
#  PPO2 stable baselines documentation: https://www.google.com/search?q=ppo2+stable+baselines&oq=ppo2+stable+baselines&aqs=chrome..69i57j69i60l3.2423j0j7&sourceid=chrome&ie=UTF-8
#
### -------------------------------------------------------------------------- ###

import numpy as np 
import configparser
import datetime
import shutil
import os
import argparse

# RL packages
from comet_ml import Experiment
import gym 
from gym import wrappers, logger, spaces
from stable_baselines import PPO2, PPO1, ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from gymfc.envs.fc_env import FlightControlEnv
from gymfc.envs.gym_env import AttitudeFlightControlEnv
import tensorflow as tf

# Training hyperparameters config file
TRAINING_CONFIG = "./gymfc/training_params.config"


# Custom callback for plotting additional values in tensorboard.
# Adds individual reward plots to help with reward engineering.
class TensorboardCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        self.env = env 
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
            
        # Log scalar value (here a random variable)
        if self.env.sim_time == self.env.max_sim_time - self.env.stepsize:
            with tf.variable_scope('max_penalty_watch', reuse=False):
                # tf.summary.scalar('mp_1', self.env.max_penalty_count_2)
                # tf.summary.scalar('mp_2', self.env.max_penalty_count_3)

                error_reward = self.env.error_reward
                signal_flux_reward = self.env.signal_flux_reward
                small_signal_reward = self.env.small_signal_reward
                saturated_max_penalty = self.env.saturated_max_penalty
                saturated_output_count = self.env.saturated_output_count
                inactive_max_penalty = self.env.inactive_max_penalty
                inactive_output_count = self.env.inactive_output_count

                summary_1 = tf.Summary(value=[tf.Summary.Value(tag='error_reward', simple_value=error_reward)])
                summary_2 = tf.Summary(value=[tf.Summary.Value(tag='signal_flux_reward', simple_value=signal_flux_reward)])
                summary_3 = tf.Summary(value=[tf.Summary.Value(tag='small_signal_reward', simple_value=small_signal_reward)])
                summary_4 = tf.Summary(value=[tf.Summary.Value(tag='saturated_max_penalty_reward', simple_value=saturated_max_penalty)])
                summary_5 = tf.Summary(value=[tf.Summary.Value(tag='saturated_output_count', simple_value=saturated_output_count)])
                summary_6 = tf.Summary(value=[tf.Summary.Value(tag='inactive_max_penalty_reward', simple_value=inactive_max_penalty)])
                summary_7 = tf.Summary(value=[tf.Summary.Value(tag='inactive_output_count', simple_value=inactive_output_count)])

                self.locals['writer'].add_summary(summary_1, self.num_timesteps)
                self.locals['writer'].add_summary(summary_2, self.num_timesteps)
                self.locals['writer'].add_summary(summary_3, self.num_timesteps)
                self.locals['writer'].add_summary(summary_4, self.num_timesteps)
                self.locals['writer'].add_summary(summary_5, self.num_timesteps)
                self.locals['writer'].add_summary(summary_6, self.num_timesteps)
                self.locals['writer'].add_summary(summary_7, self.num_timesteps)

        return True


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

    # Argument parser to select model type
    parser = argparse.ArgumentParser(description="Train a reinforcement learning flight controller.")
    parser.add_argument('-m','--model',help="RL Agent to train on.")
    args = vars(parser.parse_args())

    # Create a Comet experiment with an API key
    experiment = Experiment(api_key="Bq3mQixNCv2jVzq2YBhLdxq9A",
                            project_name="rl-flight-controller", workspace="alexbarnett12")

    # Load training parameters
    cfg = configparser.ConfigParser()
    cfg.read(TRAINING_CONFIG)
    params = cfg["PARAMETERS"]

    # Set training parameters
    learning_rate_max = float(params["learning_rate_max"])
    learning_rate_min = float(params["learning_rate_min"])
    n_steps = int(params["N_steps"])
    noptepochs = int(params["Noptepochs"])
    nminibatches = int(params["Nminibatches"])
    gamma = float(params["Gamma"])
    lam = float(params["Lam"])
    clip = float(params["Clip"])
    ent_coeff = float(params["Ent_coeff"])
    total_timesteps = int(params["Total_timesteps"])

    # Linearly decreasing learning rate (only for PPO2)
    lr_callback = create_lr_callback(learning_rate_max, learning_rate_min)

    # Report hyperparameters to Comet
    hyper_params = {"learning_rate": learning_rate_max, 
                    "steps": n_steps,
                    "epochs": noptepochs,
                    "minibatches": nminibatches,
                    "gamma": gamma,
                    "lambda": lam,
                    "clip_range": clip,
                    "ent_coeff": ent_coeff,
                    "total_timesteps": total_timesteps}
    experiment.log_parameters(hyper_params)

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
    shutil.copy("./gymfc/training_params.config", "./logs/" + model_log_dir + "/training_params.config")

    # Create a callback to save model checkpoints
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path,
                                             name_prefix='rl_model')

    # Create a separate evaluation environment
    eval_env = gym.make('attitude-fc-v0')

    # Callback to evaluate the model during training
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                                log_path=log_path, eval_freq=100000)

    # Create training environment
    env = gym.make('attitude-fc-v0')

    # Callback to add max penalty watchers to Tensorboard
    tb_callback = TensorboardCallback(env)

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback, tb_callback])

    # RL Agent; Current options are PPO1 or PPO2
    # Note: PPO2 does not work w/o vectorized environments (gymfc is not vectorized)
    if args["model"] == "PPO2":
        model = PPO2(MlpPolicy, 
                    env,
                    n_steps=n_steps,
                    learning_rate=lr_callback,
                    noptepochs=noptepochs,
                    nminibatches=nminibatches,
                    gamma=gamma,
                    lam=lam,
                    cliprange=clip,
                    ent_coef=ent_coeff,
                    tensorboard_log=tensorboard_dir)
        experiment.add_tag("PPO2")

    else:
        model = PPO1(MlpPolicy,
                     env,
                     timesteps_per_actorbatch=n_steps,
                     optim_stepsize = learning_rate_max,
                     schedule="linear",
                     optim_epochs=noptepochs,
                     optim_batchsize=nminibatches,
                     gamma=gamma,
                     lam=lam,
                     clip_param=clip,
                     entcoeff=ent_coeff,
                     tensorboard_log=tensorboard_dir)
        experiment.add_tag("PPO1")

    # Train the model. Clean up environment on user cancellation
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("INFO: Ctrl-C caught. Cleaning up...")
        env.close()
        eval_env.close()

    model.save(model_save_path)

    env.close()


if __name__ == '__main__':
    main()
   