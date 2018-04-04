import matplotlib.pyplot as plt


def plot_graphs(episode_reward, episode_length):

    x_axis = range(len(episode_length))

    plt.figure(1)
    # plt.subplot(211)
    plt.plot(x_axis, episode_length)
    plt.title('Episode Lenghts')
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Lenght")

    plt.figure(2)
    # plt.subplot(212)
    plt.plot(x_axis, episode_reward)
    plt.title('Rewards')
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
