# ! /usr/bin/env python3

from agents.base_agent import BaseAgent
from agents.randomsearch_agent import RandomSearch
from experiment import Experiment
import gym


interactive = True
env_name = "CartPole-v0"
env = gym.make(env_name)
action_space = env.action_space
agent = RandomSearch(action_space)

experiment = Experiment(env, agent)
experiment.run_randomsearch(
    max_number_of_episodes=50, interactive=interactive, display_frequency=1)
