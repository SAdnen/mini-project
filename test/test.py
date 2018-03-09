#! /usr/bin/env python3

import gym

ENV = "CartPole-v0"
EPISODES = 100


env = gym.make(ENV)

for episode in range(EPISODES):
    done = False
    state = env.reset()
    while not done:
        if state[2] > 0:
            action = 1
        else:
            action = 0
        state, reward, done, info = env.step(action)
        env.render()
