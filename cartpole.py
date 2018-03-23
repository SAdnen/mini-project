import numpy as np
import tensorflow as tf
from collections import defaultdict
import math


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

class QlearningAgent(CartPole):

    ## Learning related constants
    MIN_EXPLORE_RATE = 0.01
    MIN_LEARNING_RATE = 0.1

    def get_explore_rate(self, t, MIN_EXPLORE_RATE=MIN_EXPLORE_RATE):
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))

    def get_learning_rate(self, t, MIN_LEARNING_RATE=MIN_LEARNING_RATE):
        return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))

    def q_init(self, gamma, alpha, epsilon):
        """Initialize qlearning parameters"""
        self.Q = defaultdict(float)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def state_init(self, high, low):
        """Initialize function approximation paramaters"""
        self.num_buckets = (1, 1, 6, 3)
        self.num_actions = self.action_space.n

        self.state_bounds = list(zip(low, high))
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]

    def __init__(self, action_space,
                 high, low,
                 gamma=0.99, alpha=0.5, epsilon=0.5):
        """Initialize the agent"""

        super(QlearningAgent, self).__init__(action_space)
        self.q_init(gamma, alpha, epsilon)
        self.state_init(high, low)

    def act(self, state):


        if np.random.random()<self.epsilon:
            i = np.random.randint(0, self.num_actions)
        else:
            s1 = self.state_to_bucket(state)
            vals = [v for ((s, a), v) in self.Q.items() if s == s1]

            if len(vals) <= 0:
                i = np.random.randint(0, self.num_actions)
            else:
                i = np.argmax(vals)

            """print("vals=", vals)
            print("state=", s1)
            print("action=", i)
            print("learning rate", self.get_learning_rate(t))
            print("explore rate", self.get_explore_rate(t))"""

        return i

    def learn(self, state, action, reward, next_state):
        """Learn with qlearning"""

        s1, s2 = self.state_to_bucket(state), self.state_to_bucket(next_state)
        self.Q[(s1, action)] += 0
        for a in range(self.num_actions):
            self.Q[(s2, a)] += 0

        vals = [v for ((s, a), v) in self.Q.items() if s == s2]
        max_q = max(vals)
        td_target = reward + self.gamma * max_q
        td_delta = td_target - self.Q[(s1, action)]
        self.Q[(s1, action)] += self.alpha * td_delta
        # print(self.Q)




    def state_to_bucket(self, state):
        """Function approximation"""
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.num_buckets[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.num_buckets[i] - 1) * self.state_bounds[i][0] / bound_width
                scaling = (self.num_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)






    """    def act(self, state):
        s1 = str(state)
        vals = [v for ((s, a), v) in self.Q.items() if s == s1]
        if np.random.random() < self.epsilon:
            i = np.random.randint(0, len(self.actions))
        else:
            if len(vals) <= 0:
                i = np.random.randint(0, len(self.actions))
            else:
                i = np.argmax(vals)
        return self.actions[i]

    def learn(self, state1, action1, reward, state2, done):
        s1, a1, s2 = str(state1), str(action1), str(state2)
        self.Q[(s1, a1)] += 0
        for action in self.actions:
            a2 = str(action)
            self.Q[(s2, a2)] += 0

        vals = [v for ((s, a), v) in self.Q.items() if s == s2]
        max_q = max(vals)

        td_target = reward + self.gamma * max_q
        td_delta = td_target - self.Q[(s1, a1)]  # self.Q[(state1Str, action1)]
        self.Q[(s1, a1)] += self.alpha * td_delta"""



