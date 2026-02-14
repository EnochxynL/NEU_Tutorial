# https://blog.csdn.net/m0_73737898/article/details/135977820
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# 定义地图，0表示可通过区域，1表示障碍物
grid_map = np.array([
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1]
])

# 定义起点和终点坐标
start = (0, 0)
goal = (3, 4)

# 遗传算法参数
population_size = 20
num_generations = 100
mutation_rate = 0.2

# A*算法进行路径规划
def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))

    came_from = {}
    cost_so_far = {start: 0}

    while not open_set.empty():
        current_cost, current = open_set.get()

        if current == goal:
            break

        for neighbor in neighbors(current, grid):
            new_cost = cost_so_far[current] + 1  # 假设每个格子的代价是1

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                open_set.put((priority, neighbor))
                came_from[neighbor] = current

    # 从终点回溯找到路径
    path = []
    current = goal
    while current in came_from:
        path.insert(0, current)
        current = came_from[current]
    path.insert(0, start)

    return path
    
 # 评估函数，计算适应度
def evaluate(path, grid_map):
    for position in path:
        if grid_map[position[0]][position[1]] == 1:  # 如果路径经过障碍物
            return 0  # 适应度为零
    return 1 / len(path)  # 适应度为路径长度的倒数

# 生成初始种群
def generate_population(population_size, path):
    population = []
    for _ in range(population_size):
        mutated_path = mutate_path(path)
        population.append(mutated_path)
    return population

# 变异操作
def mutate_path(path):
    mutated_path = path.copy()
    for i in range(1, len(path) - 1):
        # 将路径点固定在格子中央
        mutated_path[i] = (
            int(np.floor((path[i][0] + path[i + 1][0]) / 2)),
            int(np.floor((path[i][1] + path[i + 1][1]) / 2))
        )
    return mutated_path

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2   
 # 可视化地图和路径
def visualize_map_and_path(grid, path):
    fig, ax = plt.subplots()

    # 绘制地图
    ax.imshow(grid, cmap='Greys', interpolation='nearest')

    # 标记起点和终点
    ax.text(start[1], start[0], 'S', color='blue', ha='center', va='center')
    ax.text(goal[1], goal[0], 'G', color='blue', ha='center', va='center')

    # 标记路径
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        ax.plot([current[1], next_node[1]], [current[0], next_node[0]], color='green', linewidth=2)

    plt.show()

# 使用A*算法初始化路径
initial_path = astar(grid_map, start, goal)

# 使用遗传算法优化路径
optimized_path = genetic_algorithm(population_size, num_generations, mutation_rate, grid_map, start, goal)

# 可视化地图和初始路径
visualize_map_and_path(grid_map, initial_path)

# 可视化地图和优化后的路径
visualize_map_and_path(grid_map, optimized_path)