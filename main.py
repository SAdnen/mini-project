# ! /usr/bin/env python3

from cartpole import CartPole
from experiment import Experiment
import gym
import numpy as np


def main():
    interactive = False
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    action_space = env.action_space
    agent = CartPole(action_space)
    experiment = Experiment(env, agent)
    experiment.run(max_number_of_episodes=100, interactive=interactive)



if __name__ == '__main__':
    main()


