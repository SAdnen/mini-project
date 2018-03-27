# ! /usr/bin/env python3

import cartpole
from experiment import Experiment
import gym


def main():
    interactive = False
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    cartople = cartpole.RandomSearch(reward_tresh=50, parameters=None)
    experiment = Experiment(env, cartople)
    experiment.run_randomsearch(interactive=True)


if __name__ == '__main__':
    main()
