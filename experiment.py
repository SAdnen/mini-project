import matplotlib.pyplot as plt
import numpy as np


class Experiment(object):

    def __init__(self, env, agent):

        self.env = env
        self.agent = agent

        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])

    def run_sarsa(self, max_number_of_episodes=100,
                  interactive=False, display_frequency=1):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # initialize state
            state = self.env.reset()

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode i.e episode length

            # repeat for each step of episode, until state is terminal
            while not done:

                # increase step counter - for display
                t += 1

                # choose action from state
                action = self.agent.act(state)

                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # choose next action from next state using policy derived from
                # Q
                next_action = self.agent.act(next_state)

                # agent learn (SARSA update)
                self.agent.learn(
                    state, action, reward, next_state, next_action)

                # state <- next state
                state = next_state

                # state <- next state, action <- next_action
                state = next_state
                action = next_action

                R += reward  # accumulate reward - for display

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()

            self.episode_length = np.append(
                self.episode_length, t)  # keep episode length - for display
            self.episode_reward = np.append(
                self.episode_reward, R)  # keep episode reward - for display

            # if interactive display, show update for the episode
            if interactive:
                self.env.render()
                self.env.close()
        # if not interactive display, show graph at the end
        if not interactive:
            episode_lengths = self.episode_length
            episode_rewards = self.episode_reward
            x_axis = range(len(episode_lengths))

            plt.figure(1)
            plt.subplot(211)
            plt.plot(x_axis, episode_lengths)
            plt.title('Episode Lenghts')
            plt.grid(True)
            plt.xlabel("Episode")
            plt.ylabel("Lenght")

            plt.figure(2)
            plt.subplot(212)
            plt.plot(x_axis, episode_rewards / np.max(np.abs(episode_rewards)))
            plt.title('Rewards')
            plt.grid(True)
            plt.xlabel("Episode")
            plt.ylabel("Reward")

            plt.show()

    def run_qlearning(self, max_number_of_episodes=100,
                      interactive=False, display_frequency=1):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # initialize state
            state = self.env.reset()

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode i.e episode length

            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1  # increase step counter - for display

                # choose action from state using policy derived from Q
                action = self.agent.act(state)

                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # agent learn (Q-Learning update)
                self.agent.learn(state, action, reward, next_state, done)

                # state <- next state
                state = next_state

                R += reward  # accumulate reward - for display

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()

            self.episode_length = np.append(
                self.episode_length, t)  # keep episode length - for display
            self.episode_reward = np.append(
                self.episode_reward, R)  # keep episode reward - for display

            # if interactive display, show update for the episode
            if interactive:
                self.env.render()
                self.env.close()
        # if not interactive display, show graph at the end
        if not interactive:
            episode_lengths = self.episode_length
            episode_rewards = self.episode_reward
            x_axis = range(len(episode_lengths))

            plt.figure(1)
            plt.subplot(211)
            plt.plot(x_axis, episode_lengths)
            plt.title('Episode Lenghts')
            plt.grid(True)
            plt.xlabel("Episode")
            plt.ylabel("Lenght")

            plt.figure(2)
            plt.subplot(212)
            plt.plot(x_axis, episode_rewards / np.max(np.abs(episode_rewards)))
            plt.title('Rewards')
            plt.grid(True)
            plt.xlabel("Episode")
            plt.ylabel("Reward")

            plt.show()
