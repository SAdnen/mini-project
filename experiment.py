import numpy as np


class Experiment(object):

    def __init__(self, env, agent):

        self.env = env
        self.agent = agent
        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])

    def fake_label(action):
        return 1

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
