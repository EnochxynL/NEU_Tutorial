# Mahdi Hassanzadeh
# Modified from https://github.com/hassanzadehmahdi/Romanian-problem-using-Astar-and-GBFS
# Map in https://github.com/GodaKartik/RomaniaRoadMapProblem/tree/main
# C++ code in https://www.freesion.com/article/9026181687/
# https://cocalc.com/share/public_paths/4bc0213188739dfe3da59b3ffd6c1c82d9eff225

import queue
from collections import deque
import matplotlib.pyplot as plt
import pprint

class DataLoader:
    DATA_PATH = "assets/romania/"

    @classmethod
    def getHeuristics(cls):
        ''' getting heuristics from file'''
        heuristics = {}
        f = open(cls.DATA_PATH+"heuristics.txt")
        for i in f.readlines():
            node_heuristic_val = i.split()
            heuristics[node_heuristic_val[0]] = int(node_heuristic_val[1])

        return heuristics
    
    @classmethod
    def getCity(cls):
        ''' getting cities location (x, y) for plotting from file'''
        city = {}
        citiesCode = {}
        f = open(cls.DATA_PATH+"cities.txt")
        j = 1
        for i in f.readlines():
            node_city_val = i.split()
            city[node_city_val[0]] = [int(node_city_val[1]), int(node_city_val[2])]

            citiesCode[j] = node_city_val[0]
            j += 1

        return city, citiesCode
    
    @classmethod
    def createGraph(cls):
        ''' creating cities graph from file'''
        graph = {}
        file = open(cls.DATA_PATH+"citiesGraph.txt")
        for i in file.readlines():
            node_val = i.split()

            if node_val[0] in graph and node_val[1] in graph:
                c = graph.get(node_val[0])
                c.append([node_val[1], node_val[2]])
                graph.update({node_val[0]: c})

                c = graph.get(node_val[1])
                c.append([node_val[0], node_val[2]])
                graph.update({node_val[1]: c})

            elif node_val[0] in graph:
                c = graph.get(node_val[0])
                c.append([node_val[1], node_val[2]])
                graph.update({node_val[0]: c})

                graph[node_val[1]] = [[node_val[0], node_val[2]]]

            elif node_val[1] in graph:
                c = graph.get(node_val[1])
                c.append([node_val[0], node_val[2]])
                graph.update({node_val[1]: c})

                graph[node_val[0]] = [[node_val[1], node_val[2]]]

            else:
                graph[node_val[0]] = [[node_val[1], node_val[2]]]
                graph[node_val[1]] = [[node_val[0], node_val[2]]]

        return graph

def BFS(startNode, graph, goalNode="Bucharest"):
    ''' Breadth First Search Algorithm '''
    queue = deque()  # 使用deque提高效率
    queue.append((startNode, [startNode]))  # 存储(当前节点, 路径)
    visited = set()

    while queue:
        current, path = queue.popleft()
        
        if current in visited:
            continue
        visited.add(current)

        if current == goalNode:
            return path

        for neighbor_info in graph[current]:
            neighbor = neighbor_info[0]
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return []  # 无路径

def DFS(startNode, graph, goalNode="Bucharest"):
    ''' Depth First Search Algorithm '''
    stack = []  # 栈：先进后出
    stack.append((startNode, [startNode]))  # 存储(当前节点, 路径)
    visited = set()

    while stack:
        current, path = stack.pop()
        
        if current in visited:
            continue
        visited.add(current)

        if current == goalNode:
            return path

        # 可选：逆序添加邻居。这会导致 DFS 算法可能返回不同的路径，虽然算法逻辑都是正确的
        # for neighbor_info in reversed(graph[current]):
        for neighbor_info in graph[current]:
            neighbor = neighbor_info[0]
            if neighbor not in visited:
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))

    return []  # 无路径

def UCS(startNode, graph, goalNode="Bucharest"):
    ''' Uniform Cost Search Algorithm. Use weight. '''
    currentSuccessor = queue.PriorityQueue()
    currentSuccessor.put((0, [startNode, 0], [startNode]))  # (累计成本, [当前节点, 边权重], 路径)
    visited = set()

    while not currentSuccessor.empty():
        cost, current, path = currentSuccessor.get()
        current_node = current[0]
        
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goalNode:
            return path
        for neighbor_info in graph[current_node]:
            neighbor = neighbor_info[0]
            edge_weight = int(neighbor_info[1])
            new_cost = cost + edge_weight
            
            if neighbor not in visited:
                new_path = path + [neighbor]
                currentSuccessor.put((new_cost, [neighbor, edge_weight], new_path))
    
    return []  # 无路径

def GBFS(startNode, heuristics, graph, goalNode="Bucharest"):
    """ Greedy Best First Search Algorithm. Use heuristics. """
    currentSuccessor = queue.PriorityQueue()
    currentSuccessor.put((heuristics[startNode], startNode, [startNode]))  # (启发值, 当前节点, 路径)
    visited = set()

    while not currentSuccessor.empty():
        h, current, path = currentSuccessor.get()
        
        if current in visited:
            continue
        visited.add(current)

        if current == goalNode:
            return path

        for neighbor_info in graph[current]:
            neighbor = neighbor_info[0]
            if neighbor not in visited:
                new_path = path + [neighbor]
                currentSuccessor.put((heuristics[neighbor], neighbor, new_path))

    return []  # 无路径

def Astar(startNode, heuristics, graph, goalNode="Bucharest"):
    ''' Astar Algorithm. Use weight + heuristics. '''
    currentSuccessor = queue.PriorityQueue()
    currentSuccessor.put((heuristics[startNode], [startNode, 0], [startNode]))  # (f值, [当前节点, 边权重], 路径)
    visited = set()

    while not currentSuccessor.empty():
        f_value, current, path = currentSuccessor.get()
        current_node = current[0]
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == goalNode:
            return path

        # 计算从起点到当前节点的实际代价 g
        g = 0
        for i in range(len(path) - 1):
            for neighbor in graph[path[i]]:
                if neighbor[0] == path[i+1]:
                    g += int(neighbor[1])
                    break

        for neighbor_info in graph[current_node]:
            neighbor = neighbor_info[0]
            edge_weight = int(neighbor_info[1])
            new_g = g + edge_weight
            new_f = new_g + heuristics[neighbor]  # f = g + h
            
            if neighbor not in visited:
                new_path = path + [neighbor]
                currentSuccessor.put((new_f, [neighbor, edge_weight], new_path))
    
    return []  # 无路径


# drawing map of answer
def drawMap(city, graph, **kwpaths):
    for i, j in city.items():
        plt.plot(j[0], j[1], "ro")
        plt.annotate(i, (j[0] + 5, j[1]))

        for k in graph[i]:
            n = city[k[0]]
            plt.plot([j[0], n[0]], [j[1], n[1]], "gray")

    color = ["red", "green", "blue", "orange", "purple", "yellow", "cyan"]

    for i, path in enumerate(kwpaths.values()):
        for j in range(len(path)):
            try:
                first = city[path[j]]
                secend = city[path[j + 1]]

                plt.plot([first[0], secend[0]], [first[1], secend[1]], color[i])
            except:
                continue

        plt.errorbar(1, 1, label=f"{list(kwpaths.keys())[i]}", color=color[i])
    plt.legend(loc="lower left")

    plt.show()


def getWeightSum(path, graph):
    weightSum = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        # 在graph[current_city]中查找next_city对应的权重
        for neighbor in graph[current_city]:
            if neighbor[0] == next_city:
                weightSum += int(neighbor[1])
                break
    return weightSum

# running the program
def main():
    heuristic = DataLoader.getHeuristics()
    graph = DataLoader.createGraph()
    pprint.pprint(graph)

    city, citiesCode = DataLoader.getCity()

    for i, j in citiesCode.items():
        print(i, j)

    while True:
        inputCode = int(input("Please enter your desired city's number (0 for exit): "))

        if inputCode == 0:
            break

        cityName = citiesCode[inputCode]

        bfs = BFS(cityName, graph)
        dfs = DFS(cityName, graph)
        print("BFS和DFS遍历图的节点顺序可能不同，这取决于图的构建方式。")
        print("BFS: ", getWeightSum(bfs, graph), " => ", bfs)
        print("DFS: ", getWeightSum(dfs, graph), " => ", dfs)
        ucs = UCS(cityName, graph)
        gbfs = GBFS(cityName, heuristic, graph)
        print("UCS: ", getWeightSum(ucs, graph), " => ", ucs)
        print("GBFS: ", getWeightSum(gbfs, graph), " => ", gbfs)
        astar = Astar(cityName, heuristic, graph)
        print("ASTAR: ", getWeightSum(astar, graph), " => ", astar)
        drawMap(city, graph, bfs=bfs, dfs=dfs, gbfs=gbfs, astar=astar, ucs=ucs)


main()