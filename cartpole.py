import numpy as np
import tensorflow as tf
from collections import defaultdict


class CartPole(object):

    """
    This Class defines the defaul behavior of CartPole agent
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        """ Returns random action from action space"""
        return self.action_space.sample()
    def learn(self):
        pass


class RandomSearch(CartPole):

    def __init__(self, reward_tresh=50, parameters=None):
        if parameters is None:
            self.reward_tresh = reward_tresh
            self.parameters = np.random(4) * 2 - 1
            self.best_parameters = []
        else:
            assert len(parameters) == 4, "length parameters should be 4"
            self.parameters = parameters

        def act(self, state):
            return self.parameters * state

        def learn(self, reward, done):
            if reward > self.reward_tresh:
                self.best_parameters.append(self.parameters)
            if done:
                self.parameters = np.random(4) * 2 - 1
