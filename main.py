# ! /usr/bin/env python3


from cartpole import CartPole, QlearningAgent
import cartpole
from experiment import Experiment
import gym



def main():
    interactive = False
    debug=True
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    action_space = env.action_space
    agent = QlearningAgent(action_space=action_space,
                           high=env.observation_space.high,
                           low=env.observation_space.low)



    experiment = Experiment(env, agent)
    experiment.run_qlearning(max_number_of_episodes=100, interactive=interactive, debug=debug)


if __name__ == '__main__':
    main()


