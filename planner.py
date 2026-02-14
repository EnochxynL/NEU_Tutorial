import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, griddata
from scipy import interpolate
import copy

def initPop(popNum, chromLength, posBound):
    pop = []
    
    # 第一代的个体初始化
    for i in range(popNum):
        # 初始化个体字典
        individual = {
            'pos': {},
            'fitness': None,
            'path': None,
            'Best': {
                'pos': {},
                'fitness': float('inf'),
                'path': None
            }
        }
        
        # 随机生成初始控制点（染色体）
        individual['pos']['x'] = (posBound[0, 1] - posBound[0, 0]) * np.random.rand(chromLength) + posBound[0, 0]
        individual['pos']['y'] = (posBound[1, 1] - posBound[1, 0]) * np.random.rand(chromLength) + posBound[1, 0]
        individual['pos']['z'] = (posBound[2, 1] - posBound[2, 0]) * np.random.rand(chromLength) + posBound[2, 0]
        
        # 将所有控制点按照x/y/z三个方向进行排序
        individual['pos']['x'] = np.sort(individual['pos']['x'])
        individual['pos']['y'] = np.sort(individual['pos']['y'])
        individual['pos']['z'] = np.sort(individual['pos']['z'])
        
        pop.append(individual)
    
    return pop

def mutation(childPop, p_mut, posBound):
    # 获取父代种群数、染色体长度
    m = len(childPop)
    n = len(childPop[0]['pos']['x'])
    
    for i in range(m):
        if np.random.rand() < p_mut:
            idx = int(round(np.random.rand() * n))
            
            # 避免越界
            if idx <= 1:
                idx = 2
            if idx == n:
                idx = n - 1
            
            # 变异：随机数替换
            childPop[i]['pos']['x'][idx-1] = np.random.rand() * (posBound[0, 1] - posBound[0, 0]) + posBound[0, 0]
            childPop[i]['pos']['y'][idx-1] = np.random.rand() * (posBound[1, 1] - posBound[1, 0]) + posBound[1, 0]
            childPop[i]['pos']['z'][idx-1] = np.random.rand() * (posBound[2, 1] - posBound[2, 0]) + posBound[2, 0]
    
    # 将所有控制点按照x/y/z三个方向进行排序
    for i in range(m):
        childPop[i]['pos']['x'] = np.sort(childPop[i]['pos']['x'])
        childPop[i]['pos']['y'] = np.sort(childPop[i]['pos']['y'])
        childPop[i]['pos']['z'] = np.sort(childPop[i]['pos']['z'])
    
    return childPop

def plotFigure(startPos, goalPos, X, Y, Z, GlobalBest):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 画起点和终点
    ax.scatter(startPos[0], startPos[1], startPos[2], s=100, c='y', marker='s', label='Start')
    ax.scatter(goalPos[0], goalPos[1], goalPos[2], s=100, c='y', marker='^', label='Goal')
    
    # 画山峰曲面
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, edgecolor='none')
    
    # 画路径和控制点
    if GlobalBest['path'] is not None:
        path = GlobalBest['path']
        pos = GlobalBest['pos']
        ax.scatter(pos['x'], pos['y'], pos['z'], c='g', marker='o', label='Control Points')
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r', linewidth=2, label='Path')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.show()

def select(pop, p_select):
    # 利用轮盘赌法执行选择操作
    fitness_values = [ind['fitness'] for ind in pop]
    fit_reverse = [1.0 / f for f in fitness_values]
    totalFit = sum(fit_reverse)
    accP = np.cumsum([f / totalFit for f in fit_reverse])  # 概率累计和
    selectNum = int(round(len(pop) * p_select))  # 选择的个体数量
    
    # 初始化父代种群
    parentPop = []
    
    for i in range(selectNum):
        # 找到比随机数大的累积概率
        rand_val = np.random.rand()
        idx = np.where(accP > rand_val)[0]
        
        if len(idx) == 0:
            # 如果找不到，选择最后一个
            parentPop.append(copy.deepcopy(pop[-1]))
        else:
            # 将首个比随机数大的累积概率的位置的个体遗传下去
            parentPop.append(copy.deepcopy(pop[idx[0]]))
    
    return parentPop

def defMap(posBound):
    mapRange = posBound[:, 1]
    
    # 初始化地形信息
    N = 10  # 山峰个数
    peaksInfo = []
    
    # 随机生成N个山峰的特征参数
    for i in range(N):
        peak = {
            'center': [mapRange[0] * (np.random.rand() * 0.8 + 0.2), 
                      mapRange[1] * (np.random.rand() * 0.8 + 0.2)],
            'height': mapRange[2] * (np.random.rand() * 0.7 + 0.3),
            'range': mapRange * 0.1 * (np.random.rand() * 0.7 + 0.3)
        }
        peaksInfo.append(peak)
    
    # 计算山峰曲面值
    peakData = np.zeros((int(mapRange[0]), int(mapRange[1])))
    
    for x in range(int(mapRange[0])):
        for y in range(int(mapRange[1])):
            total = 0
            for k in range(N):
                h_i = peaksInfo[k]['height']
                x_i = peaksInfo[k]['center'][0]
                y_i = peaksInfo[k]['center'][1]
                x_si = peaksInfo[k]['range'][0]
                y_si = peaksInfo[k]['range'][1]
                total += h_i * np.exp(-((x - x_i) / x_si) ** 2 - ((y - y_i) / y_si) ** 2)
            peakData[x, y] = total
    
    # 构造曲面网格，用于插值判断路径是否与山峰交涉
    x_coords = []
    for i in range(int(mapRange[0])):
        x_coords.extend([i] * int(mapRange[1]))
    
    y_coords = list(range(int(mapRange[1]))) * int(mapRange[0])
    z_values = peakData.flatten()
    
    # 创建网格
    xi = np.linspace(0, int(mapRange[0]) - 1, 100)
    yi = np.linspace(0, int(mapRange[1]) - 1, 100)
    X, Y = np.meshgrid(xi, yi)
    
    # 插值
    Z = griddata((x_coords, y_coords), z_values, (X, Y), method='cubic')
    
    return X, Y, Z

def crossover(parentPop, p_crs):
    # 获取父代种群数、染色体长度
    m = len(parentPop)
    n = len(parentPop[0]['pos']['x'])
    
    # 将parentPop赋值给childPop，以初始化子代种群
    childPop = copy.deepcopy(parentPop)
    
    # 交叉操作
    i = 0
    while i < m - 1:
        if np.random.rand() < p_crs:
            idx = int(round(np.random.rand() * n))
            
            # 确保索引在有效范围内
            idx = max(1, min(idx, n-1))
            
            # 交叉x坐标
            childPop[i]['pos']['x'] = np.concatenate([parentPop[i]['pos']['x'][:idx], 
                                                     parentPop[i+1]['pos']['x'][idx:]])
            childPop[i+1]['pos']['x'] = np.concatenate([parentPop[i+1]['pos']['x'][:idx], 
                                                       parentPop[i]['pos']['x'][idx:]])
            
            # 交叉y坐标
            childPop[i]['pos']['y'] = np.concatenate([parentPop[i]['pos']['y'][:idx], 
                                                     parentPop[i+1]['pos']['y'][idx:]])
            childPop[i+1]['pos']['y'] = np.concatenate([parentPop[i+1]['pos']['y'][:idx], 
                                                       parentPop[i]['pos']['y'][idx:]])
            
            # 交叉z坐标
            childPop[i]['pos']['z'] = np.concatenate([parentPop[i]['pos']['z'][:idx], 
                                                     parentPop[i+1]['pos']['z'][idx:]])
            childPop[i+1]['pos']['z'] = np.concatenate([parentPop[i+1]['pos']['z'][:idx], 
                                                       parentPop[i]['pos']['z'][idx:]])
        i += 2
    
    # 将所有控制点按照x/y/z三个方向进行排序
    for i in range(m):
        childPop[i]['pos']['x'] = np.sort(childPop[i]['pos']['x'])
        childPop[i]['pos']['y'] = np.sort(childPop[i]['pos']['y'])
        childPop[i]['pos']['z'] = np.sort(childPop[i]['pos']['z'])
    
    return childPop

def calFitness(startPos, goalPos, X, Y, Z, pop):
    for i in range(len(pop)):
        # 利用三次样条拟合散点
        x_seq = np.concatenate([[startPos[0]], pop[i]['pos']['x'], [goalPos[0]]])
        y_seq = np.concatenate([[startPos[1]], pop[i]['pos']['y'], [goalPos[1]]])
        z_seq = np.concatenate([[startPos[2]], pop[i]['pos']['z'], [goalPos[2]]])
        
        k = len(x_seq)
        i_seq = np.linspace(0, 1, k)
        I_seq = np.linspace(0, 1, 100)
        
        # 三次样条插值
        cs_x = CubicSpline(i_seq, x_seq)
        cs_y = CubicSpline(i_seq, y_seq)
        cs_z = CubicSpline(i_seq, z_seq)
        
        X_seq = cs_x(I_seq)
        Y_seq = cs_y(I_seq)
        Z_seq = cs_z(I_seq)
        path = np.column_stack((X_seq, Y_seq, Z_seq))
        
        # 判断生成的曲线是否与障碍物相交
        flag = 0
        for j in range(1, len(path)):
            x = path[j, 0]
            y = path[j, 1]
            
            # 插值获取地形高度
            if x >= X.min() and x <= X.max() and y >= Y.min() and y <= Y.max():
                # 使用griddata进行插值
                z_interp = griddata((X.flatten(), Y.flatten()), Z.flatten(), (x, y), method='cubic')
                if not np.isnan(z_interp) and path[j, 2] < z_interp:
                    flag = 1
                    break
        
        # 计算三次样条得到的离散点的路径长度（适应度）
        dx = np.diff(X_seq)
        dy = np.diff(Y_seq)
        dz = np.diff(Z_seq)
        fitness = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
        
        if flag == 1:
            pop[i]['fitness'] = 1000 * fitness
        else:
            pop[i]['fitness'] = fitness
            pop[i]['path'] = path
    
    return pop

def calBest(pop, GlobalBest):
    for i in range(len(pop)):
        # 更新个体的最优
        if pop[i]['fitness'] < pop[i]['Best']['fitness']:
            pop[i]['Best']['pos'] = copy.deepcopy(pop[i]['pos'])
            pop[i]['Best']['fitness'] = pop[i]['fitness']
            pop[i]['Best']['path'] = pop[i]['path']
        
        # 更新全局最优
        if pop[i]['Best']['fitness'] < GlobalBest['fitness']:
            GlobalBest = copy.deepcopy(pop[i]['Best'])
    
    return pop, GlobalBest

# 主程序
def main():
    # 三维路径规划模型
    startPos = np.array([1, 1, 1])
    goalPos = np.array([100, 100, 80])
    
    # 定义山峰地图
    posBound = np.array([[0, 100], [0, 100], [0, 100]])
    
    # 地图长、宽、高范围
    X, Y, Z = defMap(posBound)
    
    # 设置超参数
    chromLength = 5     # 染色体长度，代表路线的控制点数，未加首末两点
    p_select = 0.5      # 选择概率
    p_crs = 0.8         # 交叉概率
    p_mut = 0.2         # 变异概率
    popNum = 50         # 种群规模
    iterMax = 100       # 最大迭代数
    
    # 种群初始化
    # 产生初始种群
    pop = initPop(popNum, chromLength, posBound)
    
    # 计算种群适应度
    pop = calFitness(startPos, goalPos, X, Y, Z, pop)
    
    # 更新种群最优
    GlobalBest = {'pos': {}, 'fitness': float('inf'), 'path': None}
    pop, GlobalBest = calBest(pop, GlobalBest)
    
    # 主程序
    fitness_beat_iters = np.zeros(iterMax)
    
    for i in range(iterMax):
        # 选择操作
        parentPop = select(pop, p_select)
        
        # 交叉操作
        childPop = crossover(parentPop, p_crs)
        
        # 变异操作
        childPop = mutation(childPop, p_mut, posBound)
        
        # 将父代和子代组合得到新的种群
        pop = parentPop + childPop
        
        # 计算种群适应度
        pop = calFitness(startPos, goalPos, X, Y, Z, pop)
        
        # 更新种群最优
        pop, GlobalBest = calBest(pop, GlobalBest)
        
        # 把每一代的最优粒子赋值给fitness_beat_iters
        fitness_beat_iters[i] = GlobalBest['fitness']
        
        # 在命令行窗口显示每一代的信息
        print(f'第{i+1}代: 最优适应度 = {fitness_beat_iters[i]}')
        
        # 画图（每10代画一次以减少计算量）
        if (i+1) % 10 == 0 or i == iterMax-1:
            plotFigure(startPos, goalPos, X, Y, Z, GlobalBest)
    
    # 理论最小适应度：直线距离
    fitness_best = np.linalg.norm(startPos - goalPos)
    print(f'理论最优适应度 = {fitness_best}')
    
    # 画适应度迭代图
    plt.figure()
    plt.plot(fitness_beat_iters, linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('最优适应度')
    plt.title('遗传算法收敛曲线')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()