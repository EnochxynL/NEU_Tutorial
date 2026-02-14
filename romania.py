# Mahdi Hassanzadeh
# Modified from https://github.com/hassanzadehmahdi/Romanian-problem-using-Astar-and-GBFS

import queue
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
    queue = [] # 队列：先进先出
    queue.append(startNode)

    path = []

    while queue:
        current = queue.pop(0)
        path.append(current)

        if current == goalNode:
            break

        queue = []

        for i in graph[current]:
            if i[0] not in path:
                queue.append(i[0])

    return path

def DFS(startNode, graph, goalNode="Bucharest"):
    ''' Depth First Search Algorithm '''
    stack = [] # 栈：先进后出
    stack.append(startNode)

    path = []

    while stack:
        current = stack.pop()
        path.append(current)

        if current == goalNode:
            break

        stack = []

        for i in graph[current]:
            if i[0] not in path:
                stack.append(i[0])

    return path

def UCS(startNode, graph, goalNode="Bucharest"):
    ''' Uniform Cost Search Algorithm. Use weight. '''
    currentSuccessor = queue.PriorityQueue()
    weightSum = 0
    currentSuccessor.put((weightSum, [startNode, 0]))

    path = []

    while currentSuccessor.empty() == False:
        current = currentSuccessor.get()[1] # 当前节点转换为后继节点中权重最小的节点
        path.append(current[0])
        weightSum += int(current[1])
        
        if current[0] == goalNode:
            break

        currentSuccessor = queue.PriorityQueue()
        # 错误：这意味着算法只考虑当前节点的直接邻居，而不是所有可能的路径，实际上变成了一种贪心算法，而非真正的UCS。
        
        for i in graph[current[0]]:
            if i[0] not in path:
                currentSuccessor.put((int(i[1]) + weightSum, i))
        
    return path

def GBFS(startNode, heuristics, graph, goalNode="Bucharest"):
    """ Greedy Best First Search Algorithm. Use heuristics. """
    currentSuccessor = queue.PriorityQueue()
    currentSuccessor.put((heuristics[startNode], startNode)) # 当前节点“在图之外”，初始节点作为第一个后继节点（附启发值）

    path = [] # 路径记录

    while currentSuccessor.empty() == False:
        current = currentSuccessor.get()[1] # 当前节点转换为后继节点中启发值最小的节点
        path.append(current) # 将当前节点记载进入路径

        if current == goalNode:
            break

        currentSuccessor = queue.PriorityQueue() # 当前节点已转移，初始化当前节点的后继节点队列

        for i in graph[current]:
            if i[0] not in path:
                currentSuccessor.put((heuristics[i[0]], i[0])) # 将当前节点的后继节点（附启发值）记载进入当前节点的后继节点队列

    return path

def Astar(startNode, heuristics, graph, goalNode="Bucharest"):
    ''' Astar Algorithm. Use weight + heuristics. '''
    currentSuccessor = queue.PriorityQueue()
    weightSum = 0
    currentSuccessor.put((heuristics[startNode] + weightSum, [startNode, 0]))

    path = []

    while currentSuccessor.empty() == False:
        current = currentSuccessor.get()[1]
        path.append(current[0])
        weightSum += int(current[1])

        if current[0] == goalNode:
            break

        currentSuccessor = queue.PriorityQueue() # 当前节点已转移，初始化当前节点的后继节点队列
        # 错误：这意味着算法只考虑当前节点的直接邻居，而不是所有可能的路径，实际上变成了一种贪心算法，而非真正的A*。

        for i in graph[current[0]]:
            if i[0] not in path:
                currentSuccessor.put((heuristics[i[0]] + int(i[1]) + weightSum, i)) # 将当前节点的后继节点（附启发值）记载进入当前节点的后继节点队列

    return path


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