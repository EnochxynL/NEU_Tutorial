# https://github.com/Wenju-Huang/cartpole

import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()

    env.render()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()



