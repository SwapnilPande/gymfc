### -------------------------------------------------------------------------- ###
# 
#  This script evaluates a quadcopter flight controller.
#  It uses the AttitudeFlightControllerEnv gym environment that was developed to 
#  interface with the GymFC base environment.
#
#  Run this script from the root directory.
# 
### -------------------------------------------------------------------------- ###

import gym 
import argparse
from stable_baselines import PPO2
from gymfc.envs.gym_env import AttitudeFlightControlEnv
import numpy as np
import math
import matplotlib.pyplot as plt

# Plot the desired and actual angular orientations.
# threshold_percent (float): Percent of the start error
def plot_step_response(desired, actual,
                 end=1., title=None,
                 step_size=0.001, threshold_percent=0.1):

    #actual = actual[:,:end,:]
    end_time = len(desired) * step_size
    t = np.arange(0, end_time, step_size)

    #desired = desired[:end]
    threshold = threshold_percent * desired

    plot_min = -math.radians(350)
    plot_max = math.radians(350)

    subplot_index = 3
    num_subplots = 3

    f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
    f.set_size_inches(10, 5)
    if title:
        plt.suptitle(title)
    ax[0].set_xlim([0, end_time])
    res_linewidth = 2
    linestyles = ["c", "m", "b", "g"]
    reflinestyle = "k--"
    error_linestyle = "r--"

    # Always
    ax[0].set_ylabel("Roll (rad/s)")
    ax[1].set_ylabel("Pitch (rad/s)")
    ax[2].set_ylabel("Yaw (rad/s)")

    ax[-1].set_xlabel("Time (s)")

    """ ROLL """
    # Highlight the starting x axis
    ax[0].axhline(0, color="#AAAAAA")
    ax[0].plot(t, desired[:,0], reflinestyle)
    ax[0].plot(t, desired[:,0] -  threshold[:,0] , error_linestyle, alpha=0.5)
    ax[0].plot(t, desired[:,0] +  threshold[:,0] , error_linestyle, alpha=0.5)
 
    r = actual[:,0]
    ax[0].plot(t[:len(r)], r, linewidth=res_linewidth)

    ax[0].grid(True)

    """ PITCH """
    ax[1].axhline(0, color="#AAAAAA")
    ax[1].plot(t, desired[:,1], reflinestyle)
    ax[1].plot(t, desired[:,1] -  threshold[:,1] , error_linestyle, alpha=0.5)
    ax[1].plot(t, desired[:,1] +  threshold[:,1] , error_linestyle, alpha=0.5)
    p = actual[:,1]
    ax[1].plot(t[:len(p)],p, linewidth=res_linewidth)
    ax[1].grid(True)

    """ YAW """
    ax[2].axhline(0, color="#AAAAAA")
    ax[2].plot(t, desired[:,2], reflinestyle)
    ax[2].plot(t, desired[:,2] -  threshold[:,2] , error_linestyle, alpha=0.5)
    ax[2].plot(t, desired[:,2] +  threshold[:,2] , error_linestyle, alpha=0.5)
    y = actual[:,2]
    ax[2].plot(t[:len(y)],y , linewidth=res_linewidth)
    ax[2].grid(True)

    plt.show()
    
    return plt
    
    
# Evaluate the agent on a step response.
def eval(env, fc):
    actuals = [] # Store actual orientation
    desireds = [] # Store desired orientation

    # Get the initial env observation
    ob = env.reset()

    while True:
        desired = env.omega_target
        actual = env.omega_actual

        # Predict an action
        action, _states = fc.predict(ob)

        # Move in the environment
        ob, reward, done, info = env.step(action)

        actuals.append(actual)
        desireds.append(desired)

        if done:
            break
    
    env.close()

    return desireds, actuals


def main(args):

    # Create a Gym environment
    env = gym.make('attitude-fc-v0')

    # Load the agent
    if not args["file"]:
        raise Exception("Need to input a model file using -f")
    if args["file"][-4:] == ".zip":
        args["file"] = args["file"][:-4]
        
    fc = PPO2.load(args["file"])

    # Evaluate the agent
    desireds, actuals = eval(env, fc)

    # Plot the step response
    title = "Step Response"
    plt = plot_step_response(np.array(desireds), np.array(actuals), title=title)
    plt.savefig("./images/evaluation_results/step_response.jpg")


if __name__ == "__main__":

    # Parse the agent model file
    parser = argparse.ArgumentParser(description="Evaluate a flight controller and plot the step response.")
    parser.add_argument('-f','--file',help="RL Agent file")
    args = vars(parser.parse_args())

    main(args)
