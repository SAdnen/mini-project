import numpy as np
import tensorflow as tf
from collections import defaultdict


class CartPole(object):

    """
    This Class defines the defaul behavior of CartPole agent
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        """ Returns random action from action space"""
        return self.action_space.sample()


class RandomSearch(CartPole):

    def __init__(self, reward_tresh=50, parameters=None):
        if parameters:
            assert len(parameters) is 4, "lenght of parameters is not 4"
            self.parameters = parameters
        else:
            self.parameters = np.random.rand(4) * 2 - 1
        best_parameters = np.zeros(4)
        max_reward = reward_tresh
        for episode in range(max_number_of_episodes):
            state = self.env.reset()  # Initialization
            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode
            parameters = np.random.rand(4) * 2 - 1
            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1  # increase step counter - for display

                # choose action from state using policy derived from Q
                action = self.agent.act(state, parameters)

                # take action, observe reward and next state
                state, reward, done, _ = self.env.step(action)

                R += reward  # accumulate reward - for display

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()
            if R > max_reward:
                max_reward = R
                best_parameters = parameters
                print("Reward: ", R, "\n parameters:", parameters)

            self.episode_length = np.append(
                self.episode_length, t)  # keep episode length - for display
            self.episode_reward = np.append(
                self.episode_reward, R)  # keep episode reward - for disply

            # if interactive display, show update for the episode
        if interactive:
            self.env.render()
            self.env.close()
        # if not interactive display, show graph at the end
         if not interactive:
            plot_graphs(self.episode_reward, self.episode_length)
