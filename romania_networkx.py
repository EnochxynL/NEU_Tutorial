import networkx as nx
import matplotlib.pyplot as plt
import heapq  # 用于实现优先队列（GBFS）

def load_data():
    """从文件加载数据，构建 NetworkX 图，并返回城市编号映射"""
    DATA_PATH = "assets/romania/"

    G = nx.Graph()
    city_code = {}          # 编号 -> 城市名
    code = 1

    # 读取城市坐标
    with open(DATA_PATH + "cities.txt", "r") as f:
        for line in f:
            parts = line.split()
            city = parts[0]
            x, y = int(parts[1]), int(parts[2])
            G.add_node(city, pos=(x, y))
            city_code[code] = city
            code += 1

    # 读取启发式值（到 Bucharest 的估计距离）
    with open(DATA_PATH + "heuristics.txt", "r") as f:
        for line in f:
            parts = line.split()
            city = parts[0]
            h = int(parts[1])
            G.nodes[city]['heuristic'] = h

    # 读取边及其权重
    with open(DATA_PATH + "citiesGraph.txt", "r") as f:
        for line in f:
            parts = line.split()
            u, v, w = parts[0], parts[1], int(parts[2])
            G.add_edge(u, v, weight=w)

    return G, city_code

def bfs(G: nx.Graph, start, goal):
    """广度优先搜索 (BFS)，返回路径列表"""
    # 队列元素：(当前节点, 路径)
    # 使用 deque，注意先进先出
    from collections import deque
    frontier = deque([(start, [start])])
    visited = set()

    while frontier:
        current, path = frontier.popleft()
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                frontier.append((neighbor, new_path))
    return []  # 无路径

def dfs(G: nx.Graph, start, goal):
    """深度优先搜索 (DFS)，返回路径列表"""
    # 栈元素：(当前节点, 路径)
    # 使用列表模拟栈，注意先进后出
    frontier = [(start, [start])]
    visited = set()

    while frontier:
        current, path = frontier.pop()
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                frontier.append((neighbor, new_path))
    return []  # 无路径

def ucs(G: nx.Graph, start, goal):
    """统一成本搜索 (UCS)，返回路径列表"""
    # 优先队列元素：(累计成本, 当前节点, 路径)
    # 使用 heapq，注意成本小的优先
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    visited = set()

    while frontier:
        cost, current, path = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbor, edge_data in G[current].items():
            if neighbor not in visited:
                new_cost = cost + edge_data['weight']
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))
    return []  # 无路径

def gbfs(G: nx.Graph, start, goal):
    """贪婪最佳优先搜索 (GBFS)，返回路径列表"""
    # 优先队列元素：(启发值, 当前节点, 路径)
    # 使用 heapq，注意启发值小的优先
    frontier = []
    heapq.heappush(frontier, (G.nodes[start]['heuristic'], start, [start]))
    visited = set()

    while frontier:
        h, current, path = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                heapq.heappush(frontier, (G.nodes[neighbor]['heuristic'], neighbor, new_path))
    return []  # 无路径

def astar(G: nx.Graph, start, goal):
    """A* 搜索算法（手动实现）"""
    # 优先队列元素：(累计成本 + 启发值, 当前节点, 路径)
    # 使用 heapq，注意成本小的优先
    frontier = []
    heapq.heappush(frontier, (G.nodes[start]['heuristic'], start, [start]))
    visited = set()
    
    while frontier:
        cost_h, current, path = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbor, edge_data in G[current].items():
            if neighbor not in visited:
                new_cost = cost_h - G.nodes[current]['heuristic'] + edge_data['weight'] + G.nodes[neighbor]['heuristic']
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))
    return []  # 无路径

def astar_auto(G: nx.Graph, start, goal):
    """A* 搜索算法（直接使用 networkx 内置函数）"""
    # 定义启发式函数：返回目标节点的启发值（注意：标准A*启发式应依赖于当前节点和目标节点，
    # 但原数据中启发式是到Bucharest的估计，因此这里我们假设goal固定为Bucharest，
    # 直接返回目标节点的启发值（实际上是错误的，但与原代码行为保持一致）。
    # 更合理的实现应使用 G.nodes[node]['heuristic']，但原代码中的启发式是常数，不依赖于当前节点。
    # 为与原代码完全一致，我们使用一个固定函数：
    def heuristic(u, v):
        # 原代码中启发式只与目标有关，这里返回目标节点的启发值（实际上忽略了u）
        # 注意：原A*实现中优先级为 heuristics[i[0]] + int(i[1]) + distance，相当于 f = h(neighbor) + g(neighbor)
        # 因此这里返回 G.nodes[v]['heuristic'] 会导致 f = h(goal) + g，这显然不对。
        # 正确的做法是返回 G.nodes[u]['heuristic']，因为启发式是从当前节点到目标的估计。
        # 但原数据中启发式文件就是每个城市到Bucharest的估计，所以应该用当前节点的启发值。
        # 修正：返回 G.nodes[u]['heuristic']。
        return G.nodes[u]['heuristic']

    try:
        path = nx.astar_path(G, start, goal, heuristic=heuristic, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return []

def draw_map(G, **kwpaths):
    """绘制地图，高亮多条路径"""
    pos = nx.get_node_attributes(G, 'pos')

    # 绘制所有节点（灰色）
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300)
    # 绘制所有边（灰色）
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1)
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8)

    color = ["red", "green", "blue", "orange", "purple", "yellow", "cyan"]

    import matplotlib.lines as mlines
    path_lines = []

    for i, (label, path) in enumerate(kwpaths.items()):
        if path:
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color[i], width=3)
            # 高亮路径节点（与边颜色相同）
            # nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=color[i], node_size=300)
            # 高亮路径节点标签（与边颜色相同）
            # nx.draw_networkx_labels(G, pos, labels={node: node for node in path}, font_size=8, font_color=color[i])
            # 添加图例（使用代理 artist）
            path_lines.append(mlines.Line2D([], [], color=color[i], label=label))
    
    plt.legend(handles=path_lines, loc='lower left')

    plt.axis('off')
    plt.show()

def get_weight_sum(G: nx.Graph, path):
    """计算路径的总权重"""
    return sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))

def main():
    G, city_code = load_data()
    goal = "Bucharest"  # 固定目标

    # 显示城市列表
    for code, name in city_code.items():
        print(f"{code}: {name}")

    while True:
        try:
            inp = int(input("Please enter your desired city's number (0 for exit): "))
        except ValueError:
            continue

        if inp == 0:
            break
        if inp not in city_code:
            print("Invalid code, try again.")
            continue

        start = city_code[inp]
        print(f"Selected: {start}")

        bfs_path = bfs(G, start, goal)
        dfs_path = dfs(G, start, goal)
        print(f"BFS: {get_weight_sum(G, bfs_path)} =>", bfs_path)
        print(f"DFS: {get_weight_sum(G, dfs_path)} =>", dfs_path)
        print("如果图的构建方式不同（例如添加边的顺序不同），那么邻居节点的顺序就可能不同，从而导致遍历序列、生成树或找到的路径等结果出现差异。")
        ucs_path = ucs(G, start, goal)
        gbfs_path = gbfs(G, start, goal)
        print(f"UCS: {get_weight_sum(G, ucs_path)} =>", ucs_path)
        print(f"GBFS: {get_weight_sum(G, gbfs_path)} =>", gbfs_path)
        astar_path = astar(G, start, goal)
        print(f"A*: {get_weight_sum(G, astar_path)} =>", astar_path)
        astar_path_auto = astar_auto(G, start, goal)
        print(f"A* Auto: {get_weight_sum(G, astar_path_auto)} =>", astar_path_auto)

        draw_map(G, BFS=bfs_path, DFS=dfs_path, UCS=ucs_path, GBFS=gbfs_path, AStar=astar_path, AStarAuto=astar_path_auto)

if __name__ == "__main__":
    main()