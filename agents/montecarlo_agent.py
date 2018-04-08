import numpy as np
from .base_agent import BaseAgent
import math
from collections import defaultdict


class MonteCarlo(BaseAgent):

    def state_init(self, high, low):
        """Initialize function approximation paramaters"""
        self.num_buckets = (1, 1, 6, 3)
        self.num_actions = self.action_space.n

        self.state_bounds = list(zip(low, high))
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]

    def __init__(self, action_space, high, low):
        super().__init__(action_space)
        self.state_init(high, low)
        self.V = defaultdict(float)
        self.N = defaultdict(int)
        self.Q = defaultdict(float)
        self.epsilon = 0.01
        self.alpha = 0.01

    # Learning related constants
    MIN_EXPLORE_RATE = 0.1
    MIN_LEARNING_RATE = 0.01

    def get_explore_rate(self, t, MIN_EXPLORE_RATE=MIN_EXPLORE_RATE):
        return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t + 1) / 500)))

    def get_learning_rate(self, t, MIN_LEARNING_RATE=MIN_LEARNING_RATE):
        return max(MIN_LEARNING_RATE, min(0.9, 1.0 - math.log10((t + 1) / 25)))



    def act(self, state):
        """Take an action"""
        if np.random.random()<self.epsilon:
            i = np.random.randint(0, self.num_actions)
        else:
            s1 = self.state_to_bucket(state)
            vals = [v for ((s, a), v) in self.Q.items() if s == s1]

            if len(vals) <= 0:
                i = np.random.randint(0, self.num_actions)
            else:
                i = np.argmax(vals)


        return i


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


    def learn(self, states, actions, R):
        for state, action in zip(states,actions):
            s = self.state_to_bucket(state)
            self.N[s] += 1
            self.V[s] += 1/self.N[s] * (R-self.V[s])
            self.Q[s,action] += self.alpha * (R-self.Q[s,action])



        # print(self.Q)


