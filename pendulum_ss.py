import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from types import ModuleType
import sys

# ---------- 自定义空间类，模拟 gym.spaces ----------
class Discrete:
    """离散动作空间"""
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)

    def contains(self, x):
        return 0 <= x < self.n


class Box:
    """连续观测空间"""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)


class PendulumSimulator:
    """
    Pendulum 模拟器类，负责环境状态更新
    使用状态空间模型实现。
    注意，这实际上是一个非线性系统，状态空间模型只是在平衡状态下线性化近似。
    """

    def __init__(self):
        # 物理参数 (与 OpenAI Gym 的 Pendulum-v0 一致)
        self.gravity = 9.8
        self.mass = 1.0
        self.length = 1.0
        self.damping = 0.1
        self.tau = 0.05             # 时间步长 (秒)
        self.max_torque = 2.0

        # 状态初始化
        self.state = None
        
        # 初始化状态空间模型
        A_d, B_d = self.get_discrete_ss(self.tau)
        self.A_d = A_d
        self.B_d = B_d

    def get_continuous_ss(self):
        """获得连续时间状态空间模型"""
        g = self.gravity
        l = self.length
        m = self.mass
        b = self.damping
        
        # 连续状态矩阵
        A_c = np.array([
            [0, 1],
            [g/l, -b/(m*l**2)]
        ])
        B_c = np.array([[0], [1/(m*l**2)]])
        return A_c, B_c

    def get_discrete_ss(self, T=1):
        """获得离散时间状态空间模型（零阶保持器）"""
        g = self.gravity
        l = self.length
        m = self.mass
        b = self.damping
        
        # 连续状态矩阵
        A_c = np.array([
            [0, 1],
            [g/l, -b/(m*l**2)]
        ])
        B_c = np.array([[0], [1/(m*l**2)]])
        
        # 一阶欧拉近似计算离散状态矩阵
        A_d = np.eye(2) + A_c * T
        B_d = B_c * T
        
        return A_d, B_d

    def get_continuous_tf(self):
        """返回状态空间模型的传递函数"""
        try:
            import control as ctrl
            A, B = self.get_continuous_ss()
            # 创建状态空间系统
            sys_c = ctrl.ss(A, B, np.eye(2), np.zeros((2, 1)))
            # 转换为传递函数
            tf_c = ctrl.tf(sys_c)
            return tf_c
        except ImportError:
            print("Control systems library not available. Install with 'pip install control'")
            return None

    def get_discrete_tf(self):
        """返回离散状态空间模型的传递函数"""
        try:
            import control as ctrl
            # 获取正确的连续矩阵
            A_c, B_c = self.get_continuous_ss()
            # 构建连续系统
            sys_c = ctrl.ss(A_c, B_c, np.eye(2), np.zeros((2, 1)))
            # 精确离散化（零阶保持器）
            sys_d = ctrl.c2d(sys_c, self.tau, method='zoh')
            # 返回传递函数形式
            tf_d = ctrl.tf(sys_d)
            return tf_d
        except ImportError:
            print("Control systems library not available. Install with 'pip install control'")
            return None

    def setup(self):
        """重置环境，返回初始观测"""
        self.state = np.random.uniform(low=-math.pi, high=math.pi, size=(2,)).astype(np.float32)
        self.state[1] = np.random.uniform(low=-1, high=1, size=(1,)).astype(np.float32)[0]

    def loop(self, action):
        """执行动作，返回 (next_obs, reward, done, info)"""
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        
        # 直接使用离散时间状态空间模型的状态转移方程
        # x(k+1) = A_d*x(k) + B_d*u(k)
        x_vec = self.state.reshape(-1, 1)
        u_vec = np.array([torque]).reshape(-1, 1)
        
        # 计算下一个状态
        x_next = self.A_d @ x_vec + self.B_d @ u_vec
        
        # 更新状态
        theta, theta_dot = x_next.flatten()
        
        # 归一化角度到 [-pi, pi]
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
        
        self.state = np.array([theta, theta_dot], dtype=np.float32)


# ---------- Pendulum 渲染器类 ----------
class PendulumRenderer:
    """
    Pendulum 渲染器类，负责绘制环境状态
    """
    def __init__(self):
        self.fig = None
        self.ax = None
        self.pendulum_line = None
        self.bob_circle = None

    def setup(self):
        """初始化绘图窗口和元素"""
        plt.ion()                       # 交互模式
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("Pendulum")

        # 摆杆 (线段)
        self.pendulum_line, = self.ax.plot([], [], 'r-', linewidth=3)

        # 摆球 (圆形)
        self.bob_circle = patches.Circle(
            (0, 0), 0.1, linewidth=2, edgecolor='blue', facecolor='lightblue'
        )
        self.ax.add_patch(self.bob_circle)

        # 支点
        self.ax.plot(0, 0, 'ko', markersize=5)

        plt.show()

    def loop(self, theta, length):
        """更新摆的位置"""

        # 计算摆球位置
        bob_x = length * math.sin(theta)
        bob_y = -length * math.cos(theta)

        # 更新摆杆
        self.pendulum_line.set_data([0, bob_x], [0, bob_y])

        # 更新摆球
        self.bob_circle.set_center((bob_x, bob_y))

        # 刷新画布
        self.fig.canvas.draw_idle()

    def close(self):
        """关闭图形窗口"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ---------- Pendulum 环境类 ----------
class PendulumEnv:
    """
    Pendulum 摆环境
    动作：[-2.0, 2.0] 之间的连续力矩
    观测：[cos(theta), sin(theta), theta_dot]
    终止条件：无（持续运行）
    """
    def __init__(self):
        # 动作空间: 连续值 [-2, 2]
        self.action_space = Box(-2.0, 2.0, (1,), dtype=np.float32)
        # 观测空间: 3维 [cos(theta), sin(theta), theta_dot]
        high = np.array([1.0, 1.0, 8.0], dtype=np.float32)
        self.observation_space = Box(-high, high, (3,), dtype=np.float32)

        # 初始化模拟器
        self.simulator = PendulumSimulator()
        # 初始化渲染器
        self.renderer = PendulumRenderer()

    def reset(self):
        """重置环境到初始状态，返回初始观测"""
        self.simulator.setup()
        theta, theta_dot = self.simulator.state
        return np.array([math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action):
        # 动作合法性检查
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        self.simulator.loop(action)

        theta, theta_dot = self.simulator.state
        # 计算奖励
        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * action[0]**2)

        # Pendulum 环境通常不会终止
        done = False

        # 返回观测（使用 cos 和 sin 表示角度）
        obs = np.array([math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        return obs, reward, done, {}

    def render(self, mode='human'):
        """绘制当前状态"""
        theta = self.simulator.state[0]
        if self.renderer.fig is None:
            self.renderer.setup()
        self.renderer.loop(theta, self.simulator.length)
        plt.pause(0.001)   # 刷新图形

    def close(self):
        """关闭图形窗口"""
        self.renderer.close()

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)


# ---------- 模拟 gym.make 函数 ----------
def make(env_id):
    if env_id == 'Pendulum-v0':
        return PendulumEnv()
    else:
        raise ValueError(f"Unknown environment: {env_id}")


# ---------- 演示代码 ----------
if __name__ == '__main__':
    # 创建一个本地的 gym 模块，使得 import gym 和 gym.make 能工作
    gym_module = ModuleType('gym')
    gym_module.make = make
    sys.modules['gym'] = gym_module

    # 现在可以像使用 OpenAI Gym 一样调用
    import gym

    env = gym.make('Pendulum-v0')

    # 打印状态空间模型信息
    sys_c = env.simulator.get_continuous_tf()
    if sys_c is not None:
        print("连续时间传递函数:")
        print(sys_c)
    sys_d = env.simulator.get_discrete_tf()
    if sys_d is not None:
        print("离散时间传递函数:")
        print(sys_d)

    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()   # 随机动作
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(_+1))
            env.reset()
    env.close()