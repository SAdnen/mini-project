# ! /usr/bin/env python3

import cartpole
from experiment import Experiment
import gym


def main():
    INTERACTIVE = False
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    action_space = env.action_space
    agent = cartpole.RandomSearch(
        action_space, reward_tresh=50, parameters=None)
    experiment = Experiment(env, agent)
    experiment.run_randomsearch(
        max_number_of_episodes=50, interactive=INTERACTIVE)


if __name__ == '__main__':
    main()
