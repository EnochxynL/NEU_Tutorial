# https://www.cnblogs.com/xyz/p/18622600
# https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c

import gymnasium as gym
import numpy as np
 
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class PIDController:
    '''向量化的PID控制器
    实际上是四个PID控制器
    [x, x_dot, theta, theta_dot]'''
    
    # P, I, D = 0.1, 0.01, 0.5  ###
    P, I, D = [1/150, 1/950, 0.1, 0.01], [0.0005, 0.001, 0.01, 0.0001], [0.2, 0.0001, 0.5, 0.005]
    
    def __init__(self):
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0

    def setup(self):
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0

    def loop(self, error):
        self.integral += error
        self.derivative = error - self.prev_error
        self.prev_error = error
        return self.P * error + self.I * self.integral + self.D * self.derivative

env = gym.make("CartPole-v1", render_mode="human")
desired_state = np.array([0, 0, 0, 0])
desired_mask = np.array([0, 0, 1, 0])

N_episodes = 10
N_steps = 50000

pid_controller = PIDController()

for i_episode in range(N_episodes):
    state, _ = env.reset()
    pid_controller.setup()
    
    for t in range(N_steps):
        # print(f"step: {t}")
        env.render()
        error = state - desired_state
        pid = np.dot(pid_controller.loop(error), desired_mask) # 最后只使用theta的PID输出
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)
        # print(P * error + I * integral + D * derivative, pid, action)
        # print(state, action, )
 
        state, reward, done, info, _ = env.step(action)
        if done or t==N_steps-1:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()