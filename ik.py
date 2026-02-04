"""
多段机械臂粒子群优化（PSO）仿真程序，函数式编程
该程序使用粒子群算法优化机械臂关节角度，使末端执行器到达目标位置
包含交互式可视化界面，用户可点击调整目标位置
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

import time

def fitness(lengths, xg, yg, theta):
    """
    计算适应度函数（目标函数）
    适应度值定义为末端执行器位置与目标位置之间的欧氏距离平方
    
    参数:
        lengths: 各段机械臂长度数组
        xg, yg: 目标位置的x,y坐标
        theta: 各关节角度数组
    
    返回:
        res: 适应度值数组（每个粒子的适应度）
    """
    # 计算末端执行器位置：各段向量和
    x = np.sum(lengths * np.cos(theta), axis=1)
    y = np.sum(lengths * np.sin(theta), axis=1)
    # 计算与目标位置的欧氏距离平方
    res = (x - xg)**2 + (y - yg)**2
    return res


def pso(lengths, xg, yg, num_particles=100, num_iterations=50, w=0.7, c1=1.5, c2=1.5):
    """
    粒子群优化算法（PSO）实现
    优化机械臂各关节角度，使末端执行器接近目标位置
    
    参数:
        lengths: 各段机械臂长度数组
        xg, yg: 目标位置坐标
        num_particles: 粒子数量
        num_iterations: 迭代次数
        w: 惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
    
    返回:
        global_best_position: 全局最优解（最优关节角度）
    """
    # T1 = time.perf_counter()

    num_dimensions = len(lengths)  # 优化维度 = 关节数量
    # 定义搜索空间边界：每个关节角度在[0, 2π]范围内
    bounds = np.array([[0, 2*np.pi]] * len(lengths))
    
    # 初始化粒子位置和速度
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    
    # 初始化个体最优位置和适应度值
    personal_best_positions = positions.copy()
    personal_best_scores = fitness(lengths, xg, yg, positions)
    
    # 初始化全局最优位置
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    
    # PSO主循环
    for _ in range(num_iterations):
        # 生成随机数用于速度更新
        r1 = np.random.rand(num_particles, num_dimensions)
        r2 = np.random.rand(num_particles, num_dimensions)
        
        # 更新粒子速度（标准PSO速度更新公式）
        velocities = (
            w * velocities  # 惯性部分
            + c1 * r1 * (personal_best_positions - positions)  # 认知部分（个体经验）
            + c2 * r2 * (global_best_position - positions)  # 社会部分（群体经验）
        )
        
        # 更新粒子位置
        positions += velocities
        # 限制位置在边界内
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
        
        # 计算当前适应度值
        scores = fitness(lengths, xg, yg, positions)
        
        # 更新个体最优位置
        mask = scores < personal_best_scores
        personal_best_scores[mask] = scores[mask]
        personal_best_positions[mask] = positions[mask]
        
        # 更新全局最优位置
        best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_idx] < fitness(lengths, xg, yg, global_best_position.reshape(1, -1))[0]:
            global_best_position = personal_best_positions[best_idx].copy()
        
    # T2 =time.perf_counter()
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return global_best_position  # 返回全局最优解


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
    
    # 使用PSO算法计算新的最优关节角度
    theta[:] = pso(lengths, ball_pos[0], ball_pos[1])


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

    # 初始计算最优关节角度
    theta = pso(lengths, *ball_pos, num_particles, num_iterations, w, c1, c2)
    
    # 初始化全局变量
    prev_theta = np.zeros(len(lengths))  # 上一帧角度（用于动画插值）
    temp_theta = np.zeros(len(lengths))  # 插值角度
    perc = 0  # 插值百分比
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')  # 设置等比例坐标轴
    ax.set_title('Click to change ball position')  # 设置标题
    
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
        lambda f: update(f, ball, ball_pos, lines, lengths, theta, a), # XXX: 令人耳目一新的传递数据方式，函数式编程的特色
        interval=1000/60,  # 每帧间隔（毫秒）
        blit=True  # 使用blitting优化性能
    )
    
    # 绑定鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', lambda e: on_click(e, ax, ball_pos, theta))
    
    # 显示图形
    plt.show()