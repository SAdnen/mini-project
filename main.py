# ! /usr/bin/env python3

from agent import Agent
from experiment import Experiment
import gym
import numpy as np


def main():
    interactive = False
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    agent = Agent(env.action_space,
                  gamma=01.0, alpha=0.5, epsilon=0.1)
    experiment = Experiment(env, agent)
    best_parameters, list_rewards = experiment.run_randomsearch(
        max_number_of_episodes=100, interactive=interactive)
    print("Training is done")
    # [print("Parameters[{}]: {}".format(i, p))
    #  for i, p in enumerate(best_parameters)]

    best_parameters = np.array(best_parameters)
    np.save("test/best_parameters_rand2.npy", best_parameters)
    list_rewards = np.array(list_rewards)
    np.save("test/rewards_rand2.npy", list_rewards)
    mean_parameters = np.mean(best_parameters, axis=0)
    std_parameters = np.std(best_parameters, axis=0)
    print("Mean_parameters: ", mean_parameters)
    print("Standard error: ", std_parameters)

    # Run the agent using the best parameters
    interactive = True
    experiment.run_randomsearch(
        max_number_of_episodes=10, interactive=interactive,
        params=mean_parameters)


if __name__ == '__main__':
    main()
