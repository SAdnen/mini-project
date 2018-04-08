from utils.utils import plot_graphs
import numpy as np


class Experiment(object):

    def __init__(self, env, agent):

        self.env = env
        self.agent = agent
        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])

    def run(self, max_number_of_episodes=100,
            interactive=False, display_frequency=1):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # no need to initialize state
            self.env.reset()

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode

            # repeat for each step of episode, until state is terminal
            while not done:

                # increase step counter - for display
                t += 1

                # choose action from state
                action = self.agent.act()

                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # accumulate reward - for display
                R += reward

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()

            # keep episode length - for display
            self.episode_length = np.append(self.episode_length, t)

            # keep episode reward - for display
            self.episode_reward = np.append(self.episode_reward, R)

        # if interactive display, show update for the episode
        if interactive:
            self.env.close()

        plot_graphs(self.episode_reward, self.episode_length)

    def run_randomsearch(self, max_number_of_episodes=100, interactive=False,
                         display_frequency=1):
        # repeat for each episode
        for episode in range(max_number_of_episodes):
            state = self.env.reset()  # Initialization
            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode
            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1  # increase step counter - for display

                # choose action from state using policy derived from Q
                action = self.agent.act(state)

                # take action, observe reward and next state
                state, reward, done, _ = self.env.step(action)

                R += reward  # accumulate reward - for display

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()
            self.agent.learn(R, done)

            self.episode_length = np.append(
                self.episode_length, t)  # keep episode length - for display
            self.episode_reward = np.append(
                self.episode_reward, R)  # keep episode reward - for display

            # if interactive display, show update for the episode
        if interactive:
            self.env.render()
            self.env.close()

    def run_qlearning(self, max_number_of_episodes=100,
                      interactive=False, display_frequency=1, debug=True):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # initialize state
            state = self.env.reset()

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode

            # repeat for each step of episode, until state is terminal
            while not done:

                # choose action from state using policy derived from Q
                action = self.agent.act(state)
                # increment the lenght of the episode
                t += 1
                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # agent learn
                self.agent.learn(state, action, reward, next_state)

                # update state & action
                state = next_state

                R += reward  # accumulate reward - for display

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()
                    # sleep(0.1)
                if debug:
                    print("\nEpisode = %d" % episode_number)
                    print("t = %d" % t)
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("Reward: %f" % reward)

            # keep episode length - for display
            self.episode_length = np.append(self.episode_length, t)

            # keep episode reward - for display
            self.episode_reward = np.append(self.episode_reward, R)

            # update exploration rate
            self.agent.epsilon = self.agent.get_explore_rate(episode_number)

            # update learning rate
            self.agent.alpha = self.agent.get_learning_rate(episode_number)

        # if interactive display, show update for the episode
        if interactive:
            self.env.close()

        plot_graphs(self.episode_reward, self.episode_length)


    def run_montecarlo(self, max_number_of_episodes=100,
                    interactive=False, debug=False):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # no need to initialize state
            state = self.env.reset()

            # keep state tracking
            states = []

            # keep action tracking
            actions = []

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode



            # repeat for each step of episode, until state is terminal
            while not done:

                # track states
                states.append(state)

                # increase step counter - for display
                t += 1

                # choose action from state
                action = self.agent.act(state)

                # update actions
                actions.append(action)

                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # update state
                state = next_state

                # accumulate reward - for display
                R += reward

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()

                # print evolution
                if debug:
                    print("\nEpisode = %d" % episode_number)
                    print("t = %d" % t)
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("State Approximation: %s" % str(self.agent.state_to_bucket(state)))
                    print("Reward: %f" % reward)

            # learn
            self.agent.learn(states, actions, R)

            # keep episode length - for display
            self.episode_length = np.append(self.episode_length, t)

            # keep episode reward - for display
            self.episode_reward = np.append(self.episode_reward, R)

            # update exploration rate
            self.agent.epsilon = self.agent.get_explore_rate(episode_number)

            # update learning rate
            self.agent.alpha = self.agent.get_learning_rate(episode_number)

            print(self.agent.epsilon, self.agent.alpha, R)

        # if interactive display, show update for the episode
        if interactive:
            self.env.close()

        plot_graphs(self.episode_reward, self.episode_length)
