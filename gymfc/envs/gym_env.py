import numpy as np 
import configparser

import gym 
from gym import wrappers, logger, spaces
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from gymfc.envs.fc_env import FlightControlEnv

AIRCRAFT_CONFIG = "gymfc/envs/assets/gazebo/models/nf1/model.sdf"
REWARD_CONFIG = "gymfc/reward_params.config"


class AttitudeFlightControlEnv(FlightControlEnv, gym.Env):
    def __init__(self, max_sim_time=4.5):
        super(AttitudeFlightControlEnv, self).__init__(aircraft_config=AIRCRAFT_CONFIG)

        self.max_sim_time = max_sim_time
        self.num_steps = 0

        # Define the action space
        # This is currently defined for DShot ESCs that can receive a 
        # continous range of motor signals between 0 and 1
        # The dimension of the action space is dependent on the motor count
        action_space_low = np.array([0] * self.motor_count)
        action_space_high = np.array([1] * self.motor_count)
        self.action_space = spaces.Box(low=action_space_low,
                                       high=action_space_high, 
                                       dtype=np.float64)

        # Define the observation space
        # The observation space is a continuous range between -infinity and +infinity.
        self.observation_space = spaces.Box(low=np.array([-np.inf] * 6),
                                            high=np.array([np.inf] * 6),
                                            dtype=np.float64)

        # The observation and state will be initially set when reset() is called.
        # The state is a flattened array with sensor measurements in the same order
        # as defined in the model.sdf file. 
        #
        # TODO: Revisit this definition
        self.state = None

        # The observation is the angular velocity error and the change in angular velocity error
        # since the last time step
        #
        # error = omega_target - imu_angular_velocity
        # delta_error = error - error_prev
        self.observation = None

        # The current angular velocity of th UAV (roll, pitch, yaw) in rad/s.
        # This will be used to calculat the observation
        self.omega_actual = None

        # The desired angular velocity of the flight controller
        self.omega_target = [0, 0, 0]
        self.pulse_time = 0

        self.error = None # Error between desired omega and actual omega
        self.error_prev = 0 # Omega error at last time step
        self.delta_error = None # Change in error since last time step

        self.prev_action = 0

        # Load reward parameters
        cfg = configparser.ConfigParser()
        cfg.read(REWARD_CONFIG)
        params = cfg["PARAMETERS"]

        self.beta = int(params["Beta"]) # Scaling factor for penalizing large changes in control signals
        self.epsilon = float(params["Epsilon"]) # Scaling factor for the desired error gap around omega_target
        self.alpha = int(params["Alpha"]) # Scaling factor for rewarding smaller control signal values
        self.max_penalty = int(params["Max_penalty"]) # To heavily penalize for certain actions and make sure they never happen again

    # Action: Output from the neural net. 
    # The current RL algorithm implementation is from stable_baselines(https://github.com/hill-a/stable-baselines),
    # Which clips and scales the NN output to the action space defined in environment.
    # Therefore, the action can be directly passed to the environment simulation, assuming that
    # the action space has been defined to mirror the expected range of motor control signals.
    def step(self, action):
        # if np.any(np.isnan(action)):
        #     print("Fucking NaN!")
        #     action = np.array([0, 0, 0, 0])

        action_sample = self.sample_target_command(mean=action, stdev=0.1)

        # Perform a step in the simulation and receive an updated observation
        self.state = self.step_sim(action_sample)
        self.omega_actual = self.state[0:3]

        # Compute the reward for the current time step
        self.reward = self.compute_reward(action_sample)

        self.generate_command()

        # Update the observation that is fed into the neural net
        self.update_observation()

        # Check if the simulation has completed
        self.dones = self.sim_time >= self.max_sim_time
        
        # Record some info on the simulation step
        self.info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}

        self.prev_action = action 
        # self.prev_action = action if not np.any(np.isnan(action)) else np.array([0, 0, 0, 0])

        self.num_steps += 1

        return self.observation, self.reward, self.dones, self.info

        

    # Reset the environment.
    # Obtains the initial state and calculates the initial observation.
    # Generates the target angular velocity command (omega_target)
    def reset(self):
        self.num_steps = 0

        # Get the initial state
        self.state = super().reset()
        self.omega_actual = self.state[0:3]

        # Generate a target angular velocity command
        self.generate_command()

        # Calculate the initial observation
        self.update_observation()

        return self.observation
        


    # Update the observation based on the new state.
    def update_observation(self):
        
        # Calculate the error between the desired angular velocity and the current angular velocity
        self.error = (self.omega_target - self.omega_actual) / 1000

        # Calculate the change in error since the last time step
        self.delta_error = self.error - self.error_prev

        # Update the previous error as the simulation steps forward
        self.error_prev = self.error

        # The observation is an array of the error and the change in error
        self.observation = np.array([self.error, self.delta_error]).flatten()

    # Generate a desired angular velocity command 
    # This is used to calculate the angular velocity error which is sent to the agent.
    def generate_command(self):
        
        # The first 0.5 seconds are set to 0 so agent can learn its idle/hover state
        # With a 1ms sim step, this is equivalent to 500 iterations
        if self.sim_time < 0.5:
            self.omega_target = np.array([0, 0, 0])

        # Pulse ON for 2 seconds to teach acceleration to command
        elif self.sim_time < 2.5:
            self.omega_target = self.sample_target_command(mean=np.array([0, 0, 0]), stdev=100)

        # Pulse OFF for 2 seconds to teach deceleration back to idle/hover
        else:
            self.omega_target = np.array([0, 0, 0])

    
    # Sample a target command from a normal distribution.
    # The distribution has a mean of 0 deg/s and a standard deviation of
    # 100 deg/s so that aggressive maneuvers are included in the training.
    def sample_target_command(self, mean, stdev):
        return np.random.normal(loc=mean, scale=stdev)
                        

    # Calculate the reward function.
    def compute_reward(self, action):
        # if np.any(np.isnan(action)):
        #     action = np.array([0, 0, 0, 0])

        reward = 0 # Store the cumulative rewards

        # Calculate the updated error 
        error = self.omega_target - self.omega_actual
        
        # Sum of squared error for the current and previous time step
        sum_squared_error = np.sum(np.square(error))
        sum_squared_error_prev = np.sum(np.square(self.error_prev))
        reward += -(sum_squared_error - sum_squared_error_prev)

        # Penalize for large control signal fluctuations
        reward -= self.beta * np.max(np.abs(action - self.prev_action))

        # Reward smaller control signal values
        # Only apply this reward if the agent is already able to achieve an error
        # within the specified error band gap. This gap can be scaled with epsilon.
        if np.all(np.abs(error) < self.epsilon * np.abs(self.omega_target)):
            reward += self.alpha * (1 - np.mean(action))

        # Penalize for oversaturating any element in the control signal
        reward -= self.max_penalty * np.sum(np.maximum(action - 1, np.zeros(shape=action.shape)))

        # Penalize if all control signals have been saturated
        if np.all(action >= 1):
            reward -= self.max_penalty
        
        # Penalize if agent does nothing (action = 0) and there is an omega_target other than 0. 
        if np.count_nonzero(action) < 2 and np.any(self.omega_target):
            reward -= self.max_penalty

        return reward
