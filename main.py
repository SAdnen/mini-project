import numpy as np
import gym
from agent import SarsaAgent, QlearningAgent, PolicyGradientAgent
from experiment import Experiment

interactive = False
env_name = "CartPole-v0"
env = gym.make(env_name)

agent = PolicyGradientAgent(range(env.action_space.n), env.observation_space.shape[0])
experiment = Experiment(env, agent)
experiment.run_policygradient(1000, interactive)