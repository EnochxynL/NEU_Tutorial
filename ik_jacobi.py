import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

import time

class InverseKinematicsGradientDescent:
    """
    使用梯度下降法和雅可比矩阵实现机械臂逆运动学
    """
    def __init__(self, lengths):
        self.lengths = lengths
        self.num_joints = len(lengths)
    
    def forward_kinematics(self, theta):
        """
        正向运动学计算
        计算给定关节角度下的末端执行器位置
        
        Args:
            theta: 关节角度数组
            
        Returns:
            x, y: 末端执行器位置坐标
            positions: 所有关节的位置坐标（包括起点和末端）
        """
        x, y = 0, 0
        positions = [(x, y)]
        
        for i, (length, angle) in enumerate(zip(self.lengths, theta)):
            x += length * np.cos(angle)
            y += length * np.sin(angle)
            positions.append((x, y))
        
        return x, y, positions
    
    def compute_jacobian(self, theta):
        """
        计算雅可比矩阵
        J = [[dx/dθ1, dx/dθ2, ..., dx/dθn],
             [dy/dθ1, dy/dθ2, ..., dy/dθn]]
        
        Args:
            theta: 关节角度数组
            
        Returns:
            J: 雅可比矩阵 (2 x n)
        """
        x, y, positions = self.forward_kinematics(theta)
        J = np.zeros((2, self.num_joints))
        
        # 计算每个关节角度对末端位置的偏导数
        for i in range(self.num_joints):
            # 计算从第i个关节到末端的向量
            dx = x - positions[i][0]
            dy = y - positions[i][1]
            # 雅可比矩阵元素
            J[0, i] = -dy  # dx/dθi
            J[1, i] = dx   # dy/dθi
        
        return J
    
    def optimize(self, xg, yg, initial_theta=None, learning_rate=0.01, 
                 max_iterations=1000, tolerance=1e-4):
        """
        使用梯度下降法优化关节角度
        
        Args:
            xg, yg: 目标位置坐标
            initial_theta: 初始关节角度（默认随机初始化）
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
            
        Returns:
            theta: 优化后的关节角度
        """
        # 初始化关节角度
        if initial_theta is None:
            theta = np.random.uniform(0, 2*np.pi, self.num_joints)
        else:
            theta = np.array(initial_theta)
        
        # 优化循环
        for i in range(max_iterations):
            # 计算当前末端位置
            x, y, _ = self.forward_kinematics(theta)
            
            # 计算位置误差
            error = np.array([xg - x, yg - y])
            error_norm = np.linalg.norm(error)
            
            # 检查收敛条件
            if error_norm < tolerance:
                break
            
            # 计算雅可比矩阵
            J = self.compute_jacobian(theta)
            
            # 使用伪逆矩阵计算关节角度更新量
            # 当雅可比矩阵不是方阵或奇异时，使用伪逆
            J_pinv = np.linalg.pinv(J)
            delta_theta = J_pinv.dot(error)
            
            # 更新关节角度
            theta += learning_rate * delta_theta
            
            # 确保角度在[0, 2π]范围内
            theta = theta % (2*np.pi)
        
        return theta

class RoboticRunner:
    """
    机械臂可视化运行器
    """
    def __init__(self, lengths, ball_pos):
        self.lengths = lengths
        self.ball_pos = ball_pos
        self.optimizer = InverseKinematicsGradientDescent(lengths)
        
        # 初始化参数
        self.perc = 0  # 插值百分比
        self.prev_theta = np.zeros(len(lengths))  # 上一帧角度
        self.temp_theta = np.zeros(len(lengths))  # 插值角度
        
        # 初始计算最优关节角度
        self.target_theta = self.optimizer.optimize(*self.ball_pos)
        
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(10, 10))
        self.setup_axis(ax)
        self.setup_figure(fig)
        
        # 绘制初始状态
        self.lines, self.ball = self.initial_plot(ax, self.lengths, self.target_theta, self.ball_pos)
        self.axes = ax
        self.figure = fig
        
        # 创建动画
        ani = FuncAnimation(
            fig, 
            self.plot_update_callback, 
            interval=1000/60,  # 60FPS
            blit=True
        )
        self.ani = ani
        
        # 绑定鼠标点击事件
        fig.canvas.mpl_connect('button_press_event', self.on_click_callback)
    
    def setup_figure(self, fig):
        """
        设置图形属性
        """
        fig.patch.set_facecolor('#110914')
        return fig
    
    def setup_axis(self, ax):
        """
        设置坐标轴属性
        """
        ax.set_aspect('equal')
        ax.set_title('Click to change ball position (Gradient Descent)')
        ax.set_facecolor('#110914')
        
        # 设置颜色
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        return ax
    
    def initial_plot(self, ax, lengths, target_theta, ball_pos):
        """
        绘制初始状态
        """
        # 设置坐标轴范围
        total_length = sum(lengths)
        ax.set_xlim(-total_length*1.1, total_length*1.1)
        ax.set_ylim(-total_length*1.1, total_length*1.1)
        
        # 绘制地面线
        ax.plot([-total_length*0.3, total_length*0.3], [0, 0], lw=1, c='white', alpha=0.5)
        
        # 绘制目标球
        ball = Circle(ball_pos, 1, color="#55cd97")
        ax.add_patch(ball)
        
        # 绘制机械臂
        x, y = 0, 0
        lines = []
        for i, length in enumerate(lengths):
            next_x = x + length * np.cos(target_theta[i])
            next_y = y + length * np.sin(target_theta[i])
            line, = ax.plot([x, next_x], [y, next_y], 'o-', lw=3, c="#caace2")
            lines.append(line)
            x, y = next_x, next_y
        
        return lines, ball
    
    def plot_update_callback(self, frame, a=5):
        """
        动画更新函数
        """
        x, y = 0, 0
        
        # 更新插值百分比
        self.perc += (1 - self.perc)/a
        # 计算角度增量（考虑角度循环）
        delta = (self.target_theta - self.prev_theta + np.pi) % (2*np.pi) - np.pi
        # 计算插值后的角度
        self.temp_theta[:] = (self.prev_theta + self.perc * delta) % (2*np.pi)
        
        # 更新机械臂位置
        for i, line in enumerate(self.lines):
            next_x = x + self.lengths[i] * np.cos(self.temp_theta[i])
            next_y = y + self.lengths[i] * np.sin(self.temp_theta[i])
            line.set_xdata([x, next_x])
            line.set_ydata([y, next_y])
            x, y = next_x, next_y
        
        # 更新目标球位置
        self.ball.set_center((self.ball_pos[0], self.ball_pos[1]))
        
        return (*self.lines, self.ball,)
    
    def on_click_callback(self, event):
        """
        鼠标点击事件处理
        """
        if event.inaxes is None or event.inaxes != self.axes:
            return
        
        # 保存当前角度
        self.prev_theta[:] = self.target_theta
        # 重置插值百分比
        self.perc = 0
        
        # 更新目标位置
        self.ball_pos[0] = event.xdata
        self.ball_pos[1] = event.ydata
        
        # 重新优化关节角度
        self.target_theta[:] = self.optimizer.optimize(self.ball_pos[0], self.ball_pos[1], 
                                                       initial_theta=self.prev_theta)

if __name__ == '__main__':
    # 机械臂参数
    lengths = np.array([10, 8, 5, 2])
    # 初始目标位置
    ball_pos = [0, sum(lengths)*2/3]
    
    # 创建运行器并显示
    runner = RoboticRunner(lengths, ball_pos)
    plt.show()