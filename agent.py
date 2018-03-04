from collections import defaultdict
import numpy as np


class Agent(object):

    def __init__(self, action_space, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.action_space = action_space
        #  self.num_actions = len(actions) : to be removed

        self.Q = defaultdict(float)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state, parameters):
        action = np.dot(parameters, state)
        if action > 0:
            return 1
        else:
            return 0


class SarsaAgent(Agent):

    def __init__(self, action_space):
        super(SarsaAgent, self).__init__(action_space)

    def act(self, state):
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

    def learn(self, state1, action1, reward, state2, action2):
        s1, a1 = str(state1), str(action1)
        s2, a2 = str(state2), str(action2)
        self.Q[(s2, a2)] += 0
        self.Q[(s1, a1)] += 0
        self.Q[(s1, a1)] += self.alpha * \
            (reward + self.gamma * self.Q[(s2, a2)] - self.Q[(s1, a1)])


class QlearningAgent(Agent):

    def __init__(self, actions, gamma=1.0, alpha=0.5, epsilon=0.1):
        super(QlearningAgent, self).__init__(actions)
        self.Q = defaultdict(float)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state):
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
        self.Q[(s1, a1)] += self.alpha * td_delta
