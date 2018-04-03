import matplotlib.pyplot as plt


def plot_graphs(episode_reward, episode_length):
    episode_lengths = episode_length
    episode_rewards = episode_reward
    x_axis = range(len(episode_lengths))

    plt.figure(1)
    # plt.subplot(211)
    plt.plot(x_axis, episode_lengths)
    plt.title('Episode Lenghts')
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Lenght")

    plt.figure(2)
    # plt.subplot(212)
    plt.plot(x_axis, episode_rewards)
    plt.title('Rewards')
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
