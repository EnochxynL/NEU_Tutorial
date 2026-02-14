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

def gbfs(G, start, goal):
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

def astar(G, start, goal):
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

def draw_map(G, gbfs_path, astar_path):
    """绘制地图，高亮两条路径"""
    pos = nx.get_node_attributes(G, 'pos')

    # 绘制所有节点（灰色）
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300)
    # 绘制所有边（灰色）
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1)
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 高亮 GBFS 路径（绿色）
    if gbfs_path:
        gbfs_edges = list(zip(gbfs_path, gbfs_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=gbfs_edges, edge_color='green', width=3)

    # 高亮 A* 路径（蓝色）
    if astar_path:
        astar_edges = list(zip(astar_path, astar_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=astar_edges, edge_color='blue', width=3)

    # 添加图例（使用代理 artist）
    import matplotlib.lines as mlines
    green_line = mlines.Line2D([], [], color='green', label='GBFS')
    blue_line = mlines.Line2D([], [], color='blue', label='A*')
    plt.legend(handles=[green_line, blue_line], loc='lower left')

    plt.axis('off')
    plt.show()

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

        gbfs_path = gbfs(G, start, goal)
        astar_path = astar(G, start, goal)

        print("GBFS =>", gbfs_path)
        print("A*   =>", astar_path)

        draw_map(G, gbfs_path, astar_path)

if __name__ == "__main__":
    main()