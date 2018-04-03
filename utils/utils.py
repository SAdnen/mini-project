import matplotlib.pyplot as plt


class Utils(object):

    def __init__(self):
        pass

    def plot_graphs(self, episode_reward, episode_length):
        self.episode_lengths = episode_length
        self.episode_rewards = episode_reward
        x_axis = range(len(self.episode_lengths))

        plt.figure(1)
        # plt.subplot(211)
        plt.plot(x_axis, self.episode_lengths)
        plt.title('Episode Lenghts')
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Lenght")

        plt.figure(2)
        # plt.subplot(212)
        plt.plot(x_axis, self.episode_rewards)
        plt.title('Rewards')
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
