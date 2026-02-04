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


# 定义优化问题类，继承自PyMOO的Problem类
class RoboticArmProblem(Problem):
    """
    机械臂优化问题定义
    目标：找到最优关节角度，使机械臂末端与目标点距离最小
    """
    def __init__(self, lengths, xg, yg):
        """
        初始化机械臂优化问题
        
        参数:
            lengths: 各段机械臂长度数组
            xg, yg: 目标位置坐标
        """
        n_var = len(lengths)  # 变量个数 = 关节数量
        xl = np.zeros(n_var)  
        xu = 2 * np.pi * np.ones(n_var)  # 变量上界：2π
        
        # 调用父类构造函数
        super().__init__(
            n_var=n_var,  # 变量维度
            n_obj=1,  # 单目标优化
            n_constr=0,  # 无约束
            xl=xl,  # 下界
            xu=xu  # 上界
        )
        
        # 存储问题参数
        self.lengths = lengths
        self.xg = xg
        self.yg = yg
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        评估函数：计算每个解的适应度值
        
        参数:
            X: 决策变量矩阵 (n_population, n_var)
            out: 输出字典，需要填充'F'字段（目标函数值）
        """
        n_population = X.shape[0]  # 种群大小
        f_values = np.zeros(n_population)  # 存储适应度值
        
        for i in range(n_population):
            # 计算末端执行器位置
            x = np.sum(self.lengths * np.cos(X[i]))
            y = np.sum(self.lengths * np.sin(X[i]))
            
            # 计算与目标点的欧氏距离平方作为适应度值
            f_values[i] = (x - self.xg)**2 + (y - self.yg)**2
        
        # 将结果存入输出字典
        out["F"] = f_values.reshape(-1, 1)


def optimize_arm(lengths, xg, yg, num_particles=100, num_iterations=50, w=0.7, c1=1.5, c2=1.5):
    """
    使用PyMOO库的PSO算法优化机械臂关节角度
    
    参数:
        lengths: 各段机械臂长度数组
        xg, yg: 目标位置坐标
        num_particles: 粒子数量
        num_iterations: 迭代次数
        w: 惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
    
    返回:
        best_theta: 最优关节角度数组
    """
    # 创建优化问题实例
    problem = RoboticArmProblem(lengths, xg, yg)
    
    # 配置PSO算法参数
    algorithm = PSO(
        pop_size=num_particles,  # 种群大小（粒子数）
        w=w,  # 惯性权重
        c1=c1,  # 个体学习因子
        c2=c2,  # 社会学习因子
        adaptive=True,  # 自适应参数（可选）
        max_velocity_rate=0.5,  # 最大速度率（可选）
        sampling=FloatRandomSampling()  # 使用正确的采样方法
    )
    
    # 运行优化
    result = minimize(
        problem,
        algorithm,
        ('n_gen', num_iterations),  # 终止条件：迭代代数
        seed=1,  # 随机种子
        verbose=False,  # 不显示优化过程
        save_history=False  # 不保存历史记录
    )
    
    # 返回最优解
    return result.X


def update(frame, ball, ball_pos, lines, lengths, theta, a=5):
    """
    动画更新函数，每帧更新机械臂姿态和球的位置
    
    参数:
        frame: 动画帧数（未使用）
        ball: 目标球对象
        ball_pos: 目标球位置
        lines: 机械臂线段对象列表
        lengths: 机械臂各段长度
        theta: 目标关节角度
        a: 动画插值平滑参数
    
    返回:
        更新的图形对象列表
    """
    x, y = 0, 0  # 机械臂起始点（原点）
    
    # 使用全局变量实现动画插值
    global perc
    global temp_theta
    
    # 更新插值百分比（实现平滑过渡）
    perc += (1 - perc)/a
    # 计算角度增量（考虑角度循环）
    delta = (theta - prev_theta + np.pi) % (2*np.pi) - np.pi
    # 计算插值后的角度
    temp_theta[:] = (prev_theta + perc * delta) % (2*np.pi)
    
    # 更新每段机械臂的位置
    for i, line in enumerate(lines):
        # 计算当前段的末端位置
        next_x = x + lengths[i] * np.cos(temp_theta[i])
        next_y = y + lengths[i] * np.sin(temp_theta[i])
        
        # 更新线段位置
        line.set_xdata([x, next_x])
        line.set_ydata([y, next_y])
        
        # 更新起点为当前段末端，用于下一段计算
        x, y = next_x, next_y
    
    # 更新目标球位置
    ball.set_center(([ball_pos[0]], [ball_pos[1]]))
    
    # 返回需要重绘的对象
    return (*lines, ball,)


def on_click(event, axes, ball_pos, theta):
    """
    鼠标点击事件处理函数
    用户点击时更新目标位置并重新计算最优关节角度
    
    参数:
        event: 鼠标事件对象
        axes: 绘图坐标轴对象
        ball_pos: 目标球位置数组
        theta: 关节角度数组
    """
    # 确保点击在坐标轴范围内
    if event.inaxes is None or event.inaxes != axes:
        return
    
    # 使用全局变量
    global perc
    global prev_theta
    
    # 保存当前角度作为动画起始角度
    prev_theta[:] = theta
    # 重置插值百分比
    perc = 0
    
    # 更新目标球位置
    ball_pos[0] = event.xdata
    ball_pos[1] = event.ydata
    
    # 使用PyMOO的PSO算法计算新的最优关节角度
    theta[:] = optimize_arm(lengths, ball_pos[0], ball_pos[1], num_particles, num_iterations, w, c1, c2)


# 主程序入口
if __name__ == '__main__':
    
    # 机械臂参数：各段长度
    lengths = np.array([10, 8, 5, 2])
    # 初始目标位置：在总长度的2/3高度处
    ball_pos = [0, sum(lengths)*2/3]

    # PSO算法参数
    a = 5  # 动画插值平滑参数
    num_particles = 200  # 粒子数量
    num_iterations = 100  # 迭代次数
    w = 0.7  # 惯性权重
    c1 = 1.5  # 个体学习因子
    c2 = 1.5  # 社会学习因子

    # 初始计算最优关节角度（使用PyMOO）
    theta = optimize_arm(lengths, *ball_pos, num_particles, num_iterations, w, c1, c2)
    
    # 初始化全局变量
    prev_theta = np.zeros(len(lengths))  # 上一帧角度（用于动画插值）
    temp_theta = np.zeros(len(lengths))  # 插值角度
    perc = 0  # 插值百分比
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')  # 设置等比例坐标轴
    ax.set_title('Click to change ball position (PyMOO PSO)')  # 设置标题
    
    # 设置坐标轴范围（比机械臂总长度稍大）
    ax.set_xlim(-sum(lengths)*1.1, sum(lengths)*1.1)
    ax.set_ylim(-sum(lengths)*1.1, sum(lengths)*1.1)
    
    # 设置背景颜色（深色主题）
    fig.patch.set_facecolor('#110914')
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

    # 绘制地面线
    ground = ax.plot([-sum(lengths)*0.3, sum(lengths)*0.3], [0, 0], lw=1, c='white', alpha=0.5)[0]
    
    # 绘制初始机械臂
    x, y = 0, 0  # 起始点
    lines = []  # 存储线段对象的列表
    for i, length in enumerate(lengths):
        # 计算当前段末端位置
        next_x = x + length * np.cos(theta[i])
        next_y = y + length * np.sin(theta[i])
        
        # 绘制机械臂段（带关节圆点）
        line, = ax.plot([x, next_x], [y, next_y], 'o-', lw=3, c="#caace2")
        lines.append(line)
        
        # 更新起点为当前段末端
        x, y = next_x, next_y
    
    # 绘制目标球
    ball = Circle(ball_pos, 1, color="#55cd97")
    ax.add_patch(ball)
    
    # 创建动画（60FPS）
    ani = FuncAnimation(
        fig, 
        lambda f: update(f, ball, ball_pos, lines, lengths, theta, a), 
        interval=1000/60,  # 每帧间隔（毫秒）
        blit=True  # 使用blitting优化性能
    )
    
    # 绑定鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', lambda e: on_click(e, ax, ball_pos, theta))
    
    # 显示图形
    plt.show()