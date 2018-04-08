from agents.montecarlo_agent import MonteCarlo
from experiment import Experiment
import gym


interactive = False
debug = False

env_name = "CartPole-v0"
env = gym.make(env_name)
action_space = env.action_space

high=env.observation_space.high

low=env.observation_space.low
agent = MonteCarlo(action_space, high, low)

experiment = Experiment(env, agent)
experiment.run_montecarlo(max_number_of_episodes=5000, interactive=interactive, debug=debug)