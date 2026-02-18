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


class CartPoleSimulator:
    """
    CartPole 模拟器类，负责环境状态更新
    使用 Python Control Systems Library 实现
    """

    def __init__(self):
        # 物理参数 (与 OpenAI Gym 的 CartPole-v0 一致)
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.half_length = 0.5          # 杆的半长 (实际长度为 1.0)
        self.pole_mass_length = self.mass_pole * self.half_length
        self.force_mag = 10.0
        self.tau = 0.02             # 时间步长 (秒)

        # 状态初始化
        self.state = None
        
        # 初始化状态空间模型
        self._init_state_space_model()

    def _init_state_space_model(self):
        """初始化状态空间模型"""
        # 物理参数
        g = self.gravity
        M = self.mass_cart
        m = self.mass_pole
        l = self.half_length
        M_total = M + m
        
        # 计算离散时间状态空间矩阵（零阶保持器）
        # A_d = e^(A*T)
        # B_d = (∫₀^T e^(A*τ) dτ) * B
        # T = 1
        T = self.tau
        
        # 计算矩阵指数 A_d
        A_d = np.array([
            [1, T, 0, 0],
            [0, 1, (m*g*l*T)/M_total, 0],
            [0, 0, 1, T],
            [0, 0, (g*M*T)/(l*M_total), 1]
        ])
        
        # 计算 B_d（对于线性时不变系统，简化计算）
        B_d = np.array([
            [0],
            [T/M_total],
            [0],
            [-T/(l*M_total)]
        ])
        
        # 存储离散时间模型参数
        self.A_d = A_d
        self.B_d = B_d

    def setup(self):
        """重置环境，返回初始观测"""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)

    def loop(self, action):
        """执行动作，返回 (next_obs, reward, done, info)"""
        # 计算控制力
        force = self.force_mag if action == 1 else -self.force_mag
        
        # 直接使用离散时间状态空间模型的状态转移方程
        # x(k+1) = A_d*x(k) + B_d*u(k)
        x_vec = self.state.reshape(-1, 1)
        u_vec = np.array([force]).reshape(-1, 1)
        
        # 计算下一个状态
        x_next = self.A_d @ x_vec + self.B_d @ u_vec
        
        # 更新状态
        self.state = x_next.flatten().astype(np.float32)

    # 获取传递函数
    def get_transition_function(self):
        """返回状态空间模型的传递函数"""
        import control as ctrl
        # 创建离散时间状态空间系统
        sys_d = ctrl.ss(self.A_d, self.B_d, np.eye(4), np.zeros((4, 1)))
        # 转换为传递函数
        tf_d = ctrl.tf(sys_d)
        return tf_d

# ---------- CartPole 渲染器类 ----------
class CartPoleRenderer:
    """
    CartPole 渲染器类，负责绘制环境状态
    """
    def __init__(self):
        self.fig = None
        self.ax = None
        self.cart_rect = None
        self.pole_line = None

    def setup(self):
        """初始化绘图窗口和元素"""
        plt.ion()                       # 交互模式
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 4))
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("CartPole")

        # 小车 (矩形)
        cart_width = 0.4
        cart_height = 0.2
        self.cart_rect = patches.Rectangle(
            (-cart_width/2, -cart_height/2), cart_width, cart_height,
            linewidth=2, edgecolor='blue', facecolor='lightblue'
        )
        self.ax.add_patch(self.cart_rect)

        # 杆 (线段)
        self.pole_line, = self.ax.plot([], [], 'r-', linewidth=3)

        # 地面参考线
        self.ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 不 loop() 因为此时状态未知。第一次调用 render() 时会先执行 setup()，随后 loop() 传入当前状态进行绘制
        plt.show()

    def loop(self, x, theta, half_length):
        """更新小车和杆的位置"""

        # 小车中心位于 (x, 0.1) 使其底部在地面 y=0 (车高0.2)
        cart_x = x
        cart_y = 0.1
        self.cart_rect.set_xy((cart_x - 0.2, cart_y - 0.1))

        # 杆从车顶中心 (x, 0.2) 向上延伸，总长 = 2 * length
        pole_len = half_length * 2
        pole_x = [x, x + pole_len * math.sin(theta)]
        pole_y = [0.2, 0.2 + pole_len * math.cos(theta)]
        self.pole_line.set_data(pole_x, pole_y)

        # 刷新画布
        self.fig.canvas.draw_idle()

    def close(self):
        """关闭图形窗口"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ---------- CartPole 环境类 ----------
class CartPoleEnv:
    """
    CartPole 倒立摆环境
    动作：0 -> 向左推，1 -> 向右推
    观测：[车位置, 车速度, 杆角度, 杆角速度]
    终止条件：杆倾角 > ±12° 或 车位置 > ±2.4
    """
    def __init__(self):
        # 角度和位置阈值
        self.theta_threshold_radians = 12 * 2 * math.pi / 360   # 12度
        self.x_threshold = 2.4

        # 动作空间: 0或1
        self.action_space = Discrete(2)
        # 观测空间: 4维，边界使用阈值放大（实际可超出，但sample时使用该边界）
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = Box(-high, high, (4,), dtype=np.float32)

        self.steps_beyond_done = None   # 用于记录终止后调用的步数

        # 初始化模拟器
        self.simulator = CartPoleSimulator()
        # 初始化渲染器
        self.renderer = CartPoleRenderer()

    def reset(self):
        """重置环境到初始状态，返回初始观测"""
        self.simulator.setup()
        self.steps_beyond_done = None
        return np.array(self.simulator.state, dtype=np.float32)

    def step(self, action):
        # 动作合法性检查
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        self.simulator.loop(action)

        x, x_dot, theta, theta_dot = self.simulator.state
        # 判断是否终止
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # 奖励计算 (遵循 gym 逻辑)
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # 刚刚失败，给予最后一个奖励1，并记录步数
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            # 失败后继续调用 step
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.simulator.state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        """绘制当前状态"""
        x, theta = self.simulator.state[0], self.simulator.state[2]
        if self.renderer.fig is None:
            self.renderer.setup()
        self.renderer.loop(x, theta, self.simulator.half_length)
        plt.pause(0.001)   # 刷新图形

    def close(self):
        """关闭图形窗口"""
        self.renderer.close()

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)


# ---------- 模拟 gym.make 函数 ----------
def make(env_id):
    if env_id == 'CartPole-v0':
        return CartPoleEnv()
    else:
        raise ValueError(f"Unknown environment: {env_id}")


# ---------- 演示代码 (与题目要求完全一致) ----------
if __name__ == '__main__':
    # 创建一个本地的 gym 模块，使得 import gym 和 gym.make 能工作
    gym_module = ModuleType('gym')
    gym_module.make = make
    sys.modules['gym'] = gym_module

    # 现在可以像使用 OpenAI Gym 一样调用
    import gym

    env = gym.make('CartPole-v0')

    sys_d = env.simulator.get_transition_function()
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