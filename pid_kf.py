# Inspired by https://keon.io/deep-q-learning/

import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.envs.classic_control import *
import matplotlib.patches as mpatches
from dataclasses import dataclass

class SimulationRecorder:
    def __init__(self):
        self.figure, self.ax = plt.subplots(2, 2)
    
    def record_data(self, step, x_mat, state, noisy_state):
        self.ax[0][0].plot(step, x_mat[2, 0], 'ro', markersize=1)
        self.ax[0][0].plot(step, state[0], 'go', markersize=1)
        self.ax[0][0].plot(step, noisy_state[0], 'bo', markersize=1)

        self.ax[0][1].plot(step, x_mat[3, 0], 'ro', markersize=1)
        self.ax[0][1].plot(step, state[1], 'go', markersize=1)
        self.ax[0][1].plot(step, noisy_state[1], 'bo', markersize=1)

        self.ax[1][0].plot(step, x_mat[0, 0], 'ro', markersize=1)
        self.ax[1][0].plot(step, state[2], 'go', markersize=1)
        self.ax[1][0].plot(step, noisy_state[2], 'bo', markersize=1)

        self.ax[1][1].plot(step, x_mat[1, 0], 'ro', markersize=1)
        self.ax[1][1].plot(step, state[3], 'go', markersize=1)
        self.ax[1][1].plot(step, noisy_state[3], 'bo', markersize=1)
    
    def plot_results(self):
        color = ['red', 'blue', 'green']
        labels = ['kalman filtered position', 'only use measured position', 'truth position']
        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]

        self.ax[0][0].set_title('x')
        self.ax[0][0].set_xlabel('step')
        self.ax[0][0].set_ylabel('x')
        self.ax[0][0].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)

        self.ax[0][1].set_title('x_dot')
        self.ax[0][1].set_xlabel('step')
        self.ax[0][1].set_ylabel('x/s')
        self.ax[0][1].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)

        self.ax[1][0].set_title('theta')
        self.ax[1][0].set_xlabel('step')
        self.ax[1][0].set_ylabel('rad')
        self.ax[1][0].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)

        self.ax[1][1].set_title('theta_dot')
        self.ax[1][1].set_xlabel('step')
        self.ax[1][1].set_ylabel('rad/s')
        self.ax[1][1].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

@dataclass
class KFParams:
    '''卡尔曼滤波器参数'''
    
    # 系统参数
    k1 = -1/(0.5*(4/3-0.1/1.1)*1.1)
    k2 = 1/1.1 - 0.1*0.5*k1/1.1

    f_mat = np.asmatrix([[1, 0.02, 0, 0], [0, 1.0, 0, 0], [0, 0, 1, 0.02], [0.0, 0, 0, 1]])
    b_mat = np.asmatrix([[0], [0.02*k1], [0], [0.02*k2]])
    q_mat = np.asmatrix([[0.0001, 0, 0, 0], [0, 0.0001, 0, 0], [0, 0, 0.0001, 0], [0, 0, 0, 0.0001]])
    h_mat = np.asmatrix([[0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    r_mat = np.asmatrix([[0.25, 0], [0, 0.25]])

class KFilter:
    def __init__(self, f_mat, b_mat, q_mat, h_mat, r_mat):
        self.f_mat = f_mat
        self.b_mat = b_mat
        self.q_mat = q_mat
        self.h_mat = h_mat
        self.r_mat = r_mat

    def kal_filter(self, x_mat, p_mat, z_mat, action):
        x_predict = self.f_mat * x_mat + self.b_mat * action
        p_predict = self.f_mat * p_mat * self.f_mat.T + self.q_mat
        k_num = p_predict * self.h_mat.T * np.linalg.pinv(self.h_mat * p_predict * self.h_mat.T + self.r_mat)
        x_mat = x_predict + k_num * (z_mat - self.h_mat * x_predict)
        # print(x_predict[0])
        p_mat = (np.eye(4) - k_num * self.h_mat) * p_predict
        return x_mat, p_mat

@dataclass
class PIDParams:
    '''PID参数'''
    # 控制参数
    kp_cart = 2.4 - 0.5 + 0.1 + 0.25
    kd_cart = 70 + 5 + 5 - 5 - 1.5
    ki_cart = 0.008 + 0.001 + 0.004

    kp_pole = 8 - 0.5
    kd_pole = 100 - 5
    ki_pole = 0.005

class CartPoleControl:

    def __init__(self, kp_cart, ki_cart, kd_cart, kp_pole, ki_pole, kd_pole):
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.ki_cart = ki_cart

        self.kp_pole = kp_pole
        self.kd_pole = kd_pole
        self.ki_pole = ki_pole

        self.bias_cart_1 = 0
        self.bias_pole_1 = 0

        self.pole_int = 0
        self.cart_int = 0
        self.i = 0

    def pid_cart(self, position):
        bias = position  # 这句可能有问题
        # bias=self.bias_cart_1*0.8+bias*0.2
        d_bias = bias - self.bias_cart_1
        self.cart_int += bias
        balance = self.kp_cart * bias + self.kd_cart * d_bias + self.ki_cart * self.cart_int
        self.bias_cart_1 = bias
        return balance

    def pid_pole(self, angle):
        bias = angle  # 这句可能有问题
        d_bias = bias - self.bias_pole_1
        self.pole_int += bias
        balance = -self.kp_pole * bias - self.kd_pole * d_bias - self.ki_pole * self.pole_int
        self.bias_pole_1 = bias
        return balance

    def control_output(self, control_cart, control_pole):
        if DIRECT_MAG:
            return -10*(control_pole - control_cart)
        else:
            return 1 if (control_pole - control_cart) < 0 else 0


# 定义 DIRECT_MAG 变量
DIRECT_MAG = False

# 系统配置
RANDOM_NOISE = False

env = gym.make('CartPole-v1', render_mode='human')

if __name__ == '__main__':

    # 初始化控制器
    control = CartPoleControl(PIDParams.kp_cart, PIDParams.ki_cart, PIDParams.kd_cart, PIDParams.kp_pole, PIDParams.ki_pole, PIDParams.kd_pole)

    # 初始化卡尔曼滤波器
    kf = KFilter(KFParams.f_mat, KFParams.b_mat, KFParams.q_mat, KFParams.h_mat, KFParams.r_mat)

    # 初始化状态变量
    z_mat = np.asmatrix([[0.0], [0.0]])
    x_mat = np.asmatrix([[0.0], [0.0], [0.0], [0.0]])
    p_mat = np.asmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    # 初始化记录器
    recorder = SimulationRecorder()

    rewards = 0
    state, _ = env.reset()
    x_mat[0][0] = state[2]
    x_mat[1][0] = state[3]
    x_mat[2][0] = state[0]
    x_mat[3][0] = state[1]
    noisy_state = state
    # print(x_mat)
    # print(state)
    done = False
    i = 0
    j = 0
    while (j < 1000) & (abs(state[2] < 2)) & (not done):
    # while abs(state[2] < 2):
        j = j + 1
        env.render()
        # control_pole = control.pid_pole(state[2])
        # control_cart = control.pid_cart(state[0])
        control_pole = control.pid_pole(x_mat[0, 0])
        control_cart = control.pid_cart(x_mat[2, 0])
        # control_pole = control.pid_pole(noisy_state[2])
        # control_cart = control.pid_cart(noisy_state[0])

        if RANDOM_NOISE and random.random() > 0.99:
            i = 2

        if i > 0:
            if DIRECT_MAG:
                action = 10
            else:
                action = 1
            i -= 1
        else:
            action = control.control_output(control_cart, control_pole)
        # action = 0

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        noise = np.random.normal(loc=0, scale=0.5, size=4)
        noisy_state = next_state + noise
        z_mat[0][0] = noisy_state[3]
        z_mat[1][0] = noisy_state[1]
        # x_predict = f_mat * x_mat + b_mat * action
        # # print(x_predict)
        # p_predict = f_mat * p_mat * f_mat.T + q_mat
        # k_num = p_predict * h_mat.T * np.linalg.pinv(h_mat * p_predict * h_mat.T + r_mat)
        # x_mat = x_predict + k_num * (z_mat - h_mat * x_predict)
        # p_mat = (np.eye(4) - k_num * h_mat) * p_predict
        # print(x_mat.T)
        x_mat, p_mat = kf.kal_filter(x_mat, p_mat, z_mat, action)

        state = next_state
        rewards += reward
        # print(state)
        # print(action)
        recorder.record_data(j, x_mat, state, noisy_state)
    print('total rewards:'+str(rewards))
    env.close()

    recorder.plot_results()