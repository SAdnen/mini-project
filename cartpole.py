import numpy as np


class CartPole(object):

    """
    This Class defines the defaul behavior of CartPole agent
    """

    def __init__(self):
        pass

    def act(self, action_space):
        """ Returns random action from action space"""
        return self.action_space.sample()


class RandomSearch(CartPole):

    def __init__(self, reward_tresh=50, parameters=None):
        super().__init__()
        if parameters is None:
            self.reward_tresh = reward_tresh
            self.parameters = np.random.random(4) * 2 - 1
            self.best_parameters = []
        else:
            assert len(parameters) == 4, "length parameters should be 4"
            self.parameters = parameters

    def act(self, state):
        assert len(state) == 4, "length of state for cartople shoud be 4"
        action = np.dot(self.parameters, state)

        if action > 0:
            return 1
        else:
            return 0

    def learn(self, reward, done):
        if reward > self.reward_tresh:
            self.best_parameters.append((self.parameters, reward))
        if done:
            self.parameters = np.random.random(4) * 2 - 1
