# https://github.com/Wenju-Huang/cartpole

import gymnasium as gym
import numpy as np
 
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

env = gym.make("CartPole-v1", render_mode="human")

desired_state = np.array([0, 0, 0, 0])
desired_mask = np.array([0, 0, 1, 0])
 
P, I, D = 0.1, 0.01, 0.5  ###
 
N_episodes = 10
N_steps = 50000
for i_episode in range(N_episodes):
    state, _ = env.reset()
    integral = 0
    derivative = 0
    prev_error = 0
    for t in range(N_steps):
        # print(f"step: {t}")
        env.render()
        error = state - desired_state
 
        integral += error
        derivative = error - prev_error
        prev_error = error
 
        pid = np.dot(P * error + I * integral + D * derivative, desired_mask)
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)
        # print(P * error + I * integral + D * derivative, pid, action)
        # print(state, action, )
 
        state, reward, done, info, _ = env.step(action)
        if done or t==N_steps-1:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()