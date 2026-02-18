# https://www.cnblogs.com/xyz/p/18622600
# https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c

import gymnasium as gym
import numpy as np

class PIDController:
    '''向量化的PID控制器
    实际上是四个PID控制器
    [x, x_dot, theta, theta_dot]'''
    
    # P, I, D = 0.1, 0.01, 0.5  ###
    # P, I, D = [1/150, 1/950, 0.1, 0.01], [0.0005, 0.001, 0.01, 0.0001], [0.2, 0.0001, 0.5, 0.005]
    # 计算出线性近似系统的传递函数后
    # 可以使用MATLAB的Control System Designer确定PID参数
    # https://ethanr2000.medium.com/using-pid-to-cheat-an-openai-challenge-f17745226449
    # https://zhuanlan.zhihu.com/p/118543118
    # https://zhuanlan.zhihu.com/p/137231989
    # https://blog.csdn.net/qq_42249050/article/details/117749030

    def __init__(self, P=None, I=None, D=None):
        self.P = P if P is not None else [1/150, 1/950, 0.1]
        self.I = I if I is not None else [0.0005, 0.001, 0.01]
        self.D = D if D is not None else [0.2, 0.0001, 0.5]
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

def main(env=gym.make("Pendulum-v1", render_mode="human"), N_episodes=10, N_steps=50000):
    pid_controller = PIDController()

    desired_state = np.array([0, 0, 0])
    desired_mask = np.array([0, 0, 1])

    for i_episode in range(N_episodes):
        state, _ = env.reset()
        pid_controller.setup()
        
        for t in range(N_steps):
            # print(f"step: {t}")
            env.render()
            error = state - desired_state
            # pid = pid_controller.loop(error)[2] # 只使用theta的PID输出
            pid = np.dot(pid_controller.loop(error), desired_mask) # 最后只使用theta的PID输出
            def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
            action = np.array([sigmoid(pid) * 4 - 2]) # 映射到 [-2, 2]
            print(state, action, pid)
    
            state, reward, done, info, _ = env.step(action)
            if done or t==N_steps-1:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

if __name__ == "__main__":
    main()