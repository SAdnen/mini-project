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


class SarsaAgent(CartPole):

    """
    This class provides an implementation of SARASA agent for cartpole env.

    Attributes:
        act: Take state as argument and returns and action from Q table.
        learn: Update the Q table.
        Args:
            state1: state of the agent at the time step T-1.
            action1: action taken with respect to sate1.
            reward: reward assigned to action1.
            state2: new state of the agent, at the time T, after action1.
            action2: action taken with respect to state2.
    """

    def __init__(self, action_space):
        super(SarsaAgent, self).__init__(action_space)

    def act(self, state):
        s1 = str(state)
        vals = [v for ((s, a), v) in self.Q.items() if s == s1]
        if np.random.random() < self.epsilon:
            i = np.random.randint(0, len(self.action_space))
        else:
            if len(vals) <= 0:
                i = np.random.randint(0, len(self.action_space))
            else:
                i = np.argmax(vals)
        return self.action_space[i]

    def learn(self, state1, action1, reward, state2, action2):
        s1, a1 = str(state1), str(action1)
        s2, a2 = str(state2), str(action2)
        self.Q[(s2, a2)] += 0
        self.Q[(s1, a1)] += 0
        self.Q[(s1, a1)] += self.alpha * \
            (reward + self.gamma * self.Q[(s2, a2)] - self.Q[(s1, a1)])


class QlearningAgent(CartPole):

    """
    This class provides and implementation of Q-learning algorithm.

    Attributes:
        init: constuctor of the QlearningAgent class.
        Args:
            action_space: the action space of the agent.
            gamma: parameter that defines .......
            alpha: parameter that defines ......
            epsilon: parameter that defines .....

        act: take state as argument and returns action with respect to state.
        learn: update the Q-table.
        Args:
            state1: state of the agent at T-1.
            action1: action taken with respect to state1.
            reward: reward assigned to agent after commiting action1.
            state2: new state of the agent at T after taking action1.
            done: indicate whether the episode is finished or not.

    """

    def __init__(self, action_space, gamma=1.0, alpha=0.5, epsilon=0.1):
        super(QlearningAgent, self).__init__(action_space)
        self.Q = defaultdict(float)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, state):
        s1 = str(state)
        vals = [v for ((s, a), v) in self.Q.items() if s == s1]
        if np.random.random() < self.epsilon:
            i = np.random.randint(0, len(self.action_space))
        else:
            if len(vals) <= 0:
                i = np.random.randint(0, len(self.action_space))
            else:
                i = np.argmax(vals)
        return self.action_space[i]

    def learn(self, state1, action1, reward, state2, done):
        s1, a1, s2 = str(state1), str(action1), str(state2)
        self.Q[(s1, a1)] += 0
        for action in self.action_space:
            a2 = str(action)
            self.Q[(s2, a2)] += 0

        vals = [v for ((s, a), v) in self.Q.items() if s == s2]
        max_q = max(vals)

        td_target = reward + self.gamma * max_q
        td_delta = td_target - self.Q[(s1, a1)]  # self.Q[(state1Str, action1)]
        self.Q[(s1, a1)] += self.alpha * td_delta


class PolicyGradientAgent(CartPole):

    def __init__(self, action_space, obs_size, gamma=1.0, alpha=0.5, epsilon=0.1, hidden_size=128, update_frequency=10):
        super(PolicyGradientAgent, self).__init__(action_space)
        self.Q = defaultdict(float)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.update_frequency = update_frequency
        self.obs_size = obs_size

        # input_layer = tf.reshape(features["x"], [1,self.obs_size])
        self.observation = tf.placeholder("float", [None, self.obs_size])
        self.weights = {
            'h1': tf.Variable(tf.random_normal([obs_size, self.hidden_size])),
            'h2': tf.Variable(tf.random_normal([self.hidden_size, self.num_actions])),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.hidden_size])),
            'b2': tf.Variable(tf.random_normal([self.num_actions])),
        }
        self.label = tf.placeholder(tf.float32, [None, 1])
        self.return_weight = tf.placeholder(tf.float32, [None, 1])

        self.layer_1 = tf.nn.relu(
            tf.add(tf.matmul(self.observation, self.weights['h1']), self.biases['b1']))
        self.layer_2 = tf.add(
            tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2'])
        self.output = tf.nn.softmax(self.layer_2)

        self.loss = - \
            tf.reduce_mean(
                tf.log(tf.square(self.label - self.output) + 1e-04) * self.return_weight, axis=0)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def discount_rewards(self, r, gamma=0.999):
        """Take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_gradient(self):

        # tf Graph input

        pass

    def act(self, state):
        state_reshaped = np.reshape(np.array(state), (1, self.obs_size))
        y = self.sess.run(
            self.output, feed_dict={self.observation: state_reshaped})
        prob = y[0][0]
        action = 1 if np.random.uniform() < prob else 0
        print(action, y[0])
        return action  # self.action_space[0] # np.random.choice(self.action_space, p=y[0])#

    def learn(self, epx, epl, epr, episode_number):
        # Compute the discounted reward backwards through time.
        discounted_epr = self.discount_rewards(epr)
        # print(epx, epx.shape, type(epx))

        if episode_number % self.update_frequency == 0:
            arguments = {self.observation: epx, self.label:
                         epl, self.return_weight: discounted_epr}
            a = self.optimizer.run(feed_dict=arguments, session=self.sess)
            # print(a)
        # state, outputs_map = loss.forward(arguments, outputs=loss.outputs,
        # keep_for_backward=loss.outputs)

    def close(self):

        self.sess.close()
