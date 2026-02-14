# Mahdi Hassanzadeh

import queue
import matplotlib.pyplot as plt

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
        ''' getting cities location from file'''
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


def UCS(startNode, graph, goalNode="Bucharest"):
    ''' Uniform Cost Search Algorithm '''
    priorityQueue = queue.PriorityQueue()
    distance = 0
    priorityQueue.put((distance, [startNode, 0]))

    path = []

    while priorityQueue.empty() == False:
        current = priorityQueue.get()[1]
        path.append(current[0])
        distance += int(current[1])
        
        if current[0] == goalNode:
            break

        priorityQueue = queue.PriorityQueue()
        
        for i in graph[current[0]]:
            if i[0] not in path:
                priorityQueue.put((int(i[1]) + distance, i))
        
    return path

def GBFS(startNode, heuristics, graph, goalNode="Bucharest"):
    ''' Greedy Best First Search Algorithm '''
    priorityQueue = queue.PriorityQueue()
    priorityQueue.put((heuristics[startNode], startNode))

    path = []

    while priorityQueue.empty() == False:
        current = priorityQueue.get()[1]
        path.append(current)

        if current == goalNode:
            break

        priorityQueue = queue.PriorityQueue()

        for i in graph[current]:
            if i[0] not in path:
                priorityQueue.put((heuristics[i[0]], i[0]))

    return path


def Astar(startNode, heuristics, graph, goalNode="Bucharest"):
    ''' Astar Algorithm '''
    priorityQueue = queue.PriorityQueue()
    distance = 0
    priorityQueue.put((heuristics[startNode] + distance, [startNode, 0]))

    path = []

    while priorityQueue.empty() == False:
        current = priorityQueue.get()[1]
        path.append(current[0])
        distance += int(current[1])

        if current[0] == goalNode:
            break

        priorityQueue = queue.PriorityQueue()

        for i in graph[current[0]]:
            if i[0] not in path:
                priorityQueue.put((heuristics[i[0]] + int(i[1]) + distance, i))

    return path


# drawing map of answer
def drawMap(city, graph, *paths):
    for i, j in city.items():
        plt.plot(j[0], j[1], "ro")
        plt.annotate(i, (j[0] + 5, j[1]))

        for k in graph[i]:
            n = city[k[0]]
            plt.plot([j[0], n[0]], [j[1], n[1]], "gray")

    color = ["red", "green", "blue", "orange", "purple", "yellow", "cyan"]

    for path in paths:
        for i in range(len(path)):
            try:
                first = city[path[i]]
                secend = city[path[i + 1]]

                plt.plot([first[0], secend[0]], [first[1], secend[1]], color[paths.index(path)])
            except:
                continue

        plt.errorbar(1, 1, label=f"{paths.index(path)}", color=color[paths.index(path)])
    plt.legend(loc="lower left")

    plt.show()


# running the program
def main():
    heuristic = DataLoader.getHeuristics()
    graph = DataLoader.createGraph()
    city, citiesCode = DataLoader.getCity()

    for i, j in citiesCode.items():
        print(i, j)

    while True:
        inputCode = int(input("Please enter your desired city's number (0 for exit): "))

        if inputCode == 0:
            break

        cityName = citiesCode[inputCode]

        ucs = UCS(cityName, graph)
        gbfs = GBFS(cityName, heuristic, graph)
        astar = Astar(cityName, heuristic, graph)
        print("GBFS => ", gbfs)
        print("ASTAR => ", astar)
        print("UCS => ", ucs)
        drawMap(city, graph, gbfs, astar, ucs)


main()
