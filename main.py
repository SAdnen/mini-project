# ! /usr/bin/env python3

from agent import Agent
from experiment import Experiment
import gym


def main():
    interactive = True
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    agent = Agent(env.action_space,
                  gamma=01.0, alpha=0.5, epsilon=0.1)
    experiment = Experiment(env, agent)
    experiment.run_random(max_number_of_episodes=200, interactive=interactive)


if __name__ == '__main__':
    main()
