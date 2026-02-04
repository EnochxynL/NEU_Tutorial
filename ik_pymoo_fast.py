"""
多段机械臂粒子群优化（PSO）仿真程序 - 使用PyMOO库版本
该程序使用PyMOO库的粒子群算法优化机械臂关节角度，使末端执行器到达目标位置
包含交互式可视化界面，用户可点击调整目标位置
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# 导入PyMOO库
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import time

# 使用更简单的接口
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem  # 使用逐元素评估，更高效
from pymoo.optimize import minimize


# 使用ElementwiseProblem提高效率
class InverseKinematicsProblem(ElementwiseProblem):
    """
    机械臂优化问题定义 - 使用逐元素评估提高效率
    """
    def __init__(self, lengths, xg, yg):
        '''
        Docstring for __init__
        
        :param self: Description
        :param lengths: 各段机械臂长度数组。
        :param xg: 目标位置坐标
        :param yg: 目标位置坐标
        '''
        n_obj = 1 # 单目标优化
        n_var = len(lengths) # 变量个数 = 关节数量
        xl = np.zeros(n_var) # 变量下界：0
        xu = 2 * np.pi * np.ones(n_var) # 变量上界：2π
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj, 
            n_constr=0,
            xl=xl,
            xu=xu
        )
        
        self.lengths = lengths
        self.xg = xg
        self.yg = yg
    
    #callbackmethod
    def _evaluate(self, x, out, *args, **kwargs):
        """
        评估回调函数——正向动力学
        逐元素评估，避免循环
        """
        # 向量化计算末端位置
        x_pos = np.sum(self.lengths * np.cos(x))
        y_pos = np.sum(self.lengths * np.sin(x))
        
        # 计算适应度
        out["F"] = (x_pos - self.xg)**2 + (y_pos - self.yg)**2

class InverseKinematicsOptimizer:

    def __init__(self):
        # 配置PSO算法参数
        self.algorithm = PSO(
            pop_size=100,  # 种群大小（粒子数）
            w=0.7,  # 惯性权重
            c1=1.5,  # 个体学习因子
            c2=1.5,  # 社会学习因子
            adaptive=True,  # 自适应参数（可选）
            max_velocity_rate=0.5,  # 最大速度率（可选）
            sampling=FloatRandomSampling()  # 使用正确的采样方法
        )

    def optimize_arm(self, lengths, xg, yg, num_iterations=50):
        """
        使用PyMOO库的PSO算法优化机械臂关节角度
        
        Args:
            lengths: 各段机械臂长度数组
            xg, yg: 目标位置坐标
            num_particles: 粒子数量
            num_iterations: 迭代次数
        
        Returns:
            result.X: 最优关节角度数组
        """
        # 创建优化问题实例
        problem = InverseKinematicsProblem(lengths, xg, yg)
        
        # 运行优化
        result = minimize(
            problem,
            self.algorithm,
            ('n_gen', num_iterations),  # 终止条件：迭代代数
            seed=1,  # 随机种子
            verbose=False,  # 不显示优化过程
            save_history=False  # 不保存历史记录
        )
        
        # 返回最优解
        return result.X


class RoboticRunner:

    @staticmethod
    def setup_figure(fig):
        # 设置背景颜色（深色主题）
        fig.patch.set_facecolor('#110914')
        return fig

    @staticmethod
    def setup_axis(ax):
        ax.set_aspect('equal')  # 设置等比例坐标轴
        ax.set_title('Click to change ball position (PyMOO PSO)')  # 设置标题
        
        ax.set_facecolor('#110914')
        
        # 设置坐标轴和文本颜色为白色
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        return ax

    @staticmethod
    def initial_plot(ax, lengths, target_theta, ball_pos):
        
        # 设置坐标轴范围（比机械臂总长度稍大）
        ax.set_xlim(-sum(lengths)*1.1, sum(lengths)*1.1)
        ax.set_ylim(-sum(lengths)*1.1, sum(lengths)*1.1)

        # 绘制地面线
        ground = ax.plot([-sum(lengths)*0.3, sum(lengths)*0.3], [0, 0], lw=1, c='white', alpha=0.5)[0]
        
        # 绘制目标球
        ball = Circle(ball_pos, 1, color="#55cd97")
        ax.add_patch(ball)

        # 绘制初始机械臂
        x, y = 0, 0  # 起始点
        lines = []  # 存储线段对象的列表
        for i, length in enumerate(lengths):
            # 计算当前段末端位置
            next_x = x + length * np.cos(target_theta[i])
            next_y = y + length * np.sin(target_theta[i])
            
            # 绘制机械臂段（带关节圆点）
            line, = ax.plot([x, next_x], [y, next_y], 'o-', lw=3, c="#caace2")
            lines.append(line)
            
            # 更新起点为当前段末端
            x, y = next_x, next_y
        
        return lines, ball

    def plot_update_callback(self, frame, a=5): # , ball, ball_pos, lines, lengths, theta, a=5):
        """
        动画更新函数，每帧更新机械臂姿态和球的位置
        
        Args:
            frame: 动画帧数（未使用）
            a: 动画插值平滑参数
        
        Returns:
            更新的图形对象列表
        """
        x, y = 0, 0  # 机械臂起始点（原点）
        
        # 更新插值百分比（实现平滑过渡）
        self.perc += (1 - self.perc)/a
        # 计算角度增量（考虑角度循环）
        delta = (self.target_theta - self.prev_theta + np.pi) % (2*np.pi) - np.pi
        # 计算插值后的角度
        self.temp_theta[:] = (self.prev_theta + self.perc * delta) % (2*np.pi)
        
        # 更新每段机械臂的位置
        for i, line in enumerate(self.lines):
            # 计算当前段的末端位置
            next_x = x + self.lengths[i] * np.cos(self.temp_theta[i])
            next_y = y + self.lengths[i] * np.sin(self.temp_theta[i])
            
            # 更新线段位置
            line.set_xdata([x, next_x])
            line.set_ydata([y, next_y])
            
            # 更新起点为当前段末端，用于下一段计算
            x, y = next_x, next_y
        
        # 更新目标球位置
        self.ball.set_center(([self.ball_pos[0]], [self.ball_pos[1]]))
        
        # 返回需要重绘的对象
        return (*self.lines, self.ball,)
        
    def on_click_callback(self, event):
        """
        鼠标点击事件处理函数
        用户点击时更新目标位置并重新计算最优关节角度
        
        Args:
            event: 鼠标事件对象
        """
        # 确保点击在坐标轴范围内
        if event.inaxes is None or event.inaxes != self.axes:
            return
        
        # 保存当前角度作为动画起始角度
        self.prev_theta[:] = self.target_theta
        # 重置插值百分比
        self.perc = 0
        
        # 更新目标球位置
        self.ball_pos[0] = event.xdata
        self.ball_pos[1] = event.ydata
        
        # 使用PyMOO的PSO算法计算新的最优关节角度
        self.target_theta[:] = self.optimizer.optimize_arm(self.lengths, self.ball_pos[0], self.ball_pos[1])

    def __init__(self, lengths, ball_pos):
        '''
        Docstring for __init__
        
        :param self: Description
        :param lengths: 机械臂各段长度
        :param ball_pos: 目标球位置
        '''
        self.lengths = lengths
        self.ball_pos = ball_pos
        self.optimizer = InverseKinematicsOptimizer()

        self.perc = 0 # 插值百分比
        self.prev_theta = np.zeros(len(lengths)) # 上一帧角度（用于动画插值）
        self.temp_theta = np.zeros(len(lengths)) # 插值角度

        self.target_theta = self.optimizer.optimize_arm(self.lengths, *self.ball_pos) # 初始计算最优关节目标角度（使用PyMOO）

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_axis(ax)
        self.setup_figure(fig)

        self.lines, self.ball = self.initial_plot(ax, self.lengths, self.target_theta, self.ball_pos) # ball: 目标球对象 # lines: 机械臂线段对象列表
        self.axes = ax
        self.figure = fig
        
        # 创建动画（60FPS）
        ani = FuncAnimation(
            fig, 
            self.plot_update_callback, 
            interval=1000/60,  # 每帧间隔（毫秒）
            blit=True  # 使用blitting优化性能
        )
        self.ani = ani # XXX: ani对象不被调用，但是必须加入成员。否则离开__init__后会被销毁，机械臂会消失
        
        # 绑定鼠标点击事件
        fig.canvas.mpl_connect('button_press_event', self.on_click_callback)
        
# 主程序入口
if __name__ == '__main__':
    
    # 机械臂参数：各段长度
    lengths = np.array([10, 8, 5, 2])
    # 初始目标位置：在总长度的2/3高度处
    ball_pos = [0, sum(lengths)*2/3]

    runner = RoboticRunner(lengths, ball_pos)
    plt.show() # FIXME: 我不喜欢plt.show()显示图形，用fig.show可以指定对象。可惜fig.show不是阻塞的
