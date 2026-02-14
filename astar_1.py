import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class RomaniaMap:
    # 顶点名称（与 C++ 中索引一致）
    cities = [
        "Arad", 
        "Zerind", 
        "Oradea", 
        "Timisoara", 
        "Lugoj", 
        "Mehadia",
        "Drobeta", 
        "Sibiu", 
        "Rimnicu-Vilcea", 
        "Craiova", 
        "Fagaras",
        "Pitesti", 
        "Bucharest"
    ]

    map_matrix = np.array([
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
        [ 75,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
        [  0,  71,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
        [118,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
        [  0,   0,   0, 111,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
        [  0,   0,   0,   0,  70,   0,   0,   0,   0,   0,   0,   0,   0,],
        [  0,   0,   0,   0,   0,  75,   0,   0,   0,   0,   0,   0,   0,],
        [140,   0, 151,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,],
        [  0,   0,   0,   0,   0,   0,   0,  80,   0,   0,   0,   0,   0,],
        [  0,   0,   0,   0,   0,   0, 120,   0, 146,   0,   0,   0,   0,],
        [  0,   0,   0,   0,   0,   0,   0,  99,   0,   0,   0,   0,   0,],
        [  0,   0,   0,   0,   0,   0,   0,   0,  97, 138,   0,   0,   0,],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 211, 101,   0,],
        ], dtype=int)
    map_matrix = np.maximum(map_matrix, map_matrix.T)

    # 使用 from_numpy_matrix 创建无向图（矩阵非零元素变为权重）
    romania = nx.from_numpy_array(map_matrix)

    dist_to_goal = {
        "Arad": 366,
        "Bucharest": 0,
        "Craiova": 160,
        "Drobeta": 242,
        "Eforie": 161,
        "Fagaras": 178,
        "Giurgiu": 77,
        "Hirsova": 151,
        "Iasi": 226,
        "Lugoj": 244,
        "Mehadia": 241,
        "Neamt": 234,
        "Oradea": 380,
        "Pitesti": 98,
        "Rimnicu-Vilcea": 193,
        "Sibiu": 253,
        "Timisoara": 329,
        "Urziceni": 80,
        "Vaslui": 199,
        "Zerind": 374,
    }

if __name__ == "__main__":
    G = romania
    # 可视化位置：给出一个接近罗马尼亚相对位置的静态布局（可调整）

    plt.figure(figsize=(10,7))

    pos = {
        0: (-2,  2),   # Arad
        1: (-4,  4),   # Zerind
        2: (-6,  5),   # Oradea
        3: (-3, -1),   # Timisoara
        4: (-1, -2),   # Lugoj
        5: ( 0, -4),   # Mehadia
        6: ( 2, -6),   # Drobeta
        7: ( 0,  4),   # Sibiu
        8: ( 2,  1),   # Rimnicu-Vilcea
        9: ( 4, -3),   # Craiova
        10:( 2,  6),   # Fagaras
        11:( 4,  1),   # Pitesti
        12:( 6, -1)    # Bucharest
    }
    labels = dict(zip(range(len(cities)), cities))
    # 边权重标签（将浮点数转换为整数显示）
    edge_labels = {(u, v): int(d['weight']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#ffcc66")
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Romania Map (graph) - from_numpy_matrix")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import heapq

from romania import romania as G, cities, dist_to_goal

START_NAME = "Zerind"
GOAL_NAME = "Bucharest"
START = cities.index(START_NAME)
GOAL = cities.index(GOAL_NAME)

DEBUG = True

@dataclass
class NodeInfo:
    state: int
    parent: Optional[int]
    action: Optional[int]
    depth: int
    g: float
    h: float

def edge_cost(u: int, v: int) -> float:
    data = G.get_edge_data(u, v)
    if data is None:
        return float('inf')
    return float(data.get('weight', float('inf')))

def heuristic(state: int) -> float:
    name = cities[state]
    return float(dist_to_goal.get(name, 0))

def format_frontier(frontier: List[NodeInfo]) -> str:
    return '[' + ', '.join(f"{cities[n.state]}(g={n.g},h={n.h})" for n in frontier) + ']'

def reconstruct_path(nodes: Dict[int, NodeInfo], goal_state: int) -> List[int]:
    path = []
    cur = goal_state
    while cur is not None:
        path.append(cur)
        cur = nodes[cur].parent
    return list(reversed(path))

def format_tree_yaml(nodes: Dict[int, NodeInfo], root_state: int) -> str:
    """Create a YAML-like nested representation of the discovered search tree.

    Behavior:
      - If a node has discovered children, print the node as a mapping with
        its children nested under it.
      - If a node has no discovered children, print the node with a numeric
        value equal to its heuristic (h).

    This mirrors the example layout the user requested.
    """
    # build mapping parent -> [children]
    children_map: Dict[Optional[int], List[int]] = {}
    for state, ni in nodes.items():
        parent = ni.parent
        children_map.setdefault(parent, []).append(state)

    def _emit(s: Optional[int], indent: int) -> List[str]:
        lines: List[str] = []
        for c in sorted(children_map.get(s, []), key=lambda x: cities[x]):
            ni = nodes[c]
            name = cities[c]
            # if this node has children discovered, emit as mapping
            if children_map.get(c):
                lines.append(' ' * indent + f"{name}:")
                # emit children of c at increased indent
                lines.extend(_emit(c, indent + 4))
            else:
                # leaf: show heuristic value only (matches provided examples)
                lines.append(' ' * indent + f"{name}: {ni.h}")
        return lines

    # header: show root with its heuristic on a single line
    root_info = nodes.get(root_state, NodeInfo(root_state, None, None, 0, 0.0, heuristic(root_state)))
    out_lines: List[str] = [f"{cities[root_state]}: {root_info.h}", ""]
    # emit full tree under root (children of root_state)
    out_lines.append(f"{cities[root_state]}:")
    out_lines.extend(_emit(root_state, 4))
    return '\n'.join(out_lines) + '\n'

def draw_from_yaml(yaml_str: str, extra_text: str = '', filename: str = 'search_tree') -> None:
    """Draw the search tree from its YAML-like representation using networkx.

    Args:
        yaml_str: The YAML-like string representing the search tree.
        filename: The base filename for output (without extension).
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Parse YAML-like lines into a tree structure (parent -> [children])
    lines = yaml_str.splitlines()
    stack: List[Tuple[int, str]] = []  # (indent_level, node_name)
    children_map: Dict[str, List[str]] = {}
    seen_nodes: set = set()
    order: List[str] = []

    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        parts = raw.strip().split(':', 1)
        node_name = parts[0].strip()
        # keep insertion order for later stable layout
        if node_name not in seen_nodes:
            order.append(node_name)
            seen_nodes.add(node_name)

        # attach to parent according to indent stack
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            parent_name = stack[-1][1]
            children_map.setdefault(parent_name, [])
            if node_name not in children_map[parent_name]:
                children_map[parent_name].append(node_name)
        else:
            # ensure root exists in map
            children_map.setdefault(node_name, [])
        stack.append((indent, node_name))

    # build adjacency (ensure all nodes present)
    for n in order:
        children_map.setdefault(n, [])

    # compute tidy tree layout: x positions assigned by inorder of leaves
    pos: Dict[str, Tuple[float, float]] = {}
    next_x = [0]

    def compute_pos(n: str, depth: int):
        kids = children_map.get(n, [])
        if not kids:
            x = next_x[0]
            pos[n] = (x, -depth)
            next_x[0] += 1
            return pos[n]
        # compute positions for children
        child_positions = [compute_pos(c, depth + 1) for c in kids]
        xs = [p[0] for p in child_positions]
        x = sum(xs) / len(xs)
        pos[n] = (x, -depth)
        return pos[n]

    # find roots: nodes that are never children
    all_children = {c for kids in children_map.values() for c in kids}
    roots = [n for n in children_map.keys() if n not in all_children]
    if not roots:
        roots = list(children_map.keys())[:1]

    for r in roots:
        compute_pos(r, 0)

    # build graph for drawing edges
    G = nx.DiGraph()
    for parent, kids in children_map.items():
        G.add_node(parent)
        for c in kids:
            G.add_node(c)
            G.add_edge(parent, c)

    plt.figure(figsize=(10, max(6, len(pos) * 0.25)))
    ax = plt.gca()
    ax.set_axis_off()

    # draw edges as straight lines and nodes as circles with labels
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='gray')

    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=1200, facecolor='lightblue', edgecolor='k', zorder=3)
        ax.text(x, y, n, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)

    # draw YAML original text inside the figure (bottom-left corner) with box
    # preserve newlines so each YAML line is on its own line
    bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.02, extra_text + '\n' + yaml_str, ha='left', va='bottom', fontsize=8, transform=ax.transAxes, bbox=bbox_props)

    # adjust limits with some padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)

    plt.title("Search Tree")
    plt.savefig(f"{filename}.svg", bbox_inches='tight')
    plt.close()

    # from graphviz import Digraph

    # dot = Digraph(comment='Search Tree')
    # lines = yaml_str.strip().splitlines()

    # stack: List[Tuple[int, str]] = []  # (indent_level, node_name)
    # for line in lines:
    #     if not line.strip():
    #         continue
    #     indent = len(line) - len(line.lstrip(' '))
    #     parts = line.strip().split(':')
    #     node_name = parts[0].strip()
    #     label = node_name
    #     if len(parts) > 1 and parts[1].strip():
    #         label += f" ({parts[1].strip()})"
    #     dot.node(node_name, label=label)

    #     while stack and stack[-1][0] >= indent:
    #         stack.pop()
    #     if stack:
    #         parent_name = stack[-1][1]
    #         dot.edge(parent_name, node_name)
    #     stack.append((indent, node_name))

    # dot.render(filename, format='png', cleanup=True)

def run_bfs(debug=False):
    print('\n=== BFS ===')
    frontier: List[NodeInfo] = [NodeInfo(START, None, None, 0, 0.0, heuristic(START))]
    nodes: Dict[int, NodeInfo] = {START: frontier[0]}
    explored = set()
    step = 0
    current_newly_added = 0
    while frontier:
        node = frontier.pop(0)
        if DEBUG:
            step_yaml = f"Step {step}: frontier = {format_frontier(frontier)}"
            print(step_yaml)
            tree_yaml = format_tree_yaml(nodes, START) # pretty-print YAML-style discovered tree
            print(tree_yaml)
            remove_yaml = f"Remove: {cities[node.state]}"
            print(remove_yaml)
        if DEBUG:
            added_yaml = f"Frontier Added: {current_newly_added}"
            draw_from_yaml(tree_yaml, extra_text=step_yaml + '\n' + remove_yaml + '\n' + added_yaml, filename=f'search_tree/bfs_step_{step}')
            # append tree snapshot for later use
            with open('search_tree/bfs_steps.yaml', 'a', encoding='utf-8') as tf:
                tf.write(f"BFS Step {step}:\n{tree_yaml}\n")
        if node.state == GOAL:
            print('Goal found')
            return nodes, reconstruct_path(nodes, GOAL), step
        explored.add(node.state)
        # expand
        current_newly_added = 0
        for nbr in G.neighbors(node.state):
            # Avoid expanding nodes that would create cycles or are already
            # discovered in the frontier/explored set.
            # TODO: this check prevents cycles; record or explain in docs if
            # we want a different duplicate-handling strategy.
            if nbr in explored or any(n.state == nbr for n in frontier):
                continue
            g = node.g + edge_cost(node.state, nbr)
            ni = NodeInfo(nbr, node.state, nbr, node.depth + 1, g, heuristic(nbr))
            nodes[nbr] = ni
            frontier.append(ni)
            current_newly_added += 1
        step += 1
    return nodes, [], step

def run_dfs(debug=False, max_depth=1000):
    print('\n=== DFS ===')
    frontier: List[NodeInfo] = [NodeInfo(START, None, None, 0, 0.0, heuristic(START))]
    nodes: Dict[int, NodeInfo] = {START: frontier[0]}
    explored = set()
    step = 0
    current_newly_added = 0
    while frontier:
        node = frontier.pop()  # stack
        if DEBUG:
            step_yaml = f"Step {step}: frontier = {format_frontier(frontier)}"
            print(step_yaml)
            tree_yaml = format_tree_yaml(nodes, START) # pretty-print YAML-style discovered tree
            print(tree_yaml)
            remove_yaml = f"Remove: {cities[node.state]}"
            print(remove_yaml)
        if DEBUG:
            added_yaml = f"Frontier Added: {current_newly_added}"
            draw_from_yaml(tree_yaml, extra_text=step_yaml + '\n' + remove_yaml + '\n' + added_yaml, filename=f'search_tree/dfs_step_{step}')
            with open('search_tree/dfs_steps.yaml', 'a', encoding='utf-8') as tf:
                tf.write(f"DFS Step {step}:\n{tree_yaml}\n")
        if node.state == GOAL:
            print('Goal found')
            return nodes, reconstruct_path(nodes, GOAL), step
        explored.add(node.state)
        if node.depth >= max_depth:
            step += 1
            continue
        # expand (push neighbors in order)
        current_newly_added = 0
        for nbr in G.neighbors(node.state):
            # Avoid adding neighbors already seen (prevents cycles).
            # TODO: this is the primary guard against cycles in DFS expansion.
            if nbr in explored or any(n.state == nbr for n in frontier):
                continue
            g = node.g + edge_cost(node.state, nbr)
            ni = NodeInfo(nbr, node.state, nbr, node.depth + 1, g, heuristic(nbr))
            nodes[nbr] = ni
            frontier.append(ni)
            current_newly_added += 1
        step += 1
    return nodes, [], step

def run_ucs(debug=False):
    print('\n=== UCS ===')
    counter = 0
    frontier_heap: List[Tuple[float,int,int,NodeInfo]] = []  # (g, counter, state, node)
    start = NodeInfo(START, None, None, 0, 0.0, heuristic(START))
    heapq.heappush(frontier_heap, (start.g, counter, start.state, start)); counter += 1
    nodes: Dict[int, NodeInfo] = {START: start}
    explored = set()
    step = 0
    current_newly_added = 0
    while frontier_heap:
        frontier_list = [item[3] for item in frontier_heap]
        g, _, state, node = heapq.heappop(frontier_heap)
        if DEBUG:
            step_yaml = f"Step {step}: frontier = {format_frontier(frontier_list)}"
            print(step_yaml)
            tree_yaml = format_tree_yaml(nodes, START) # pretty-print YAML-style discovered tree
            print(tree_yaml)
            remove_yaml = f"Remove: {cities[state]} (g={g})"
            print(remove_yaml)
        if DEBUG:
            added_yaml = f"Frontier Added: {current_newly_added}"
            draw_from_yaml(tree_yaml, extra_text=step_yaml + '\n' + remove_yaml + '\n' + added_yaml, filename=f'search_tree/ucs_step_{step}')
            with open('search_tree/ucs_steps.yaml', 'a', encoding='utf-8') as tf:
                tf.write(f"UCS Step {step}:\n{tree_yaml}\n")
        if state == GOAL:
            print('Goal found')
            return nodes, reconstruct_path(nodes, GOAL), step
        if state in explored:
            step += 1
            continue
        explored.add(state)
        current_newly_added = 0
        for nbr in G.neighbors(state):
            new_g = node.g + edge_cost(state, nbr)
            # If neighbor already explored with a cheaper cost, skip it.
            # TODO: this check helps avoid revisiting nodes (cycle avoidance
            # and redundant higher-cost paths).
            if nbr in explored:
                if nodes.get(nbr) and nodes[nbr].g <= new_g:
                    continue
            if (not nodes.get(nbr)) or new_g < nodes[nbr].g:
                ni = NodeInfo(nbr, state, nbr, node.depth + 1, new_g, heuristic(nbr))
                nodes[nbr] = ni
                heapq.heappush(frontier_heap, (ni.g, counter, ni.state, ni)); counter += 1
                current_newly_added += 1
        step += 1
    return nodes, [], step

def run_greedy(debug=False):
    print('\n=== Greedy Best-First ===')
    counter = 0
    frontier_heap: List[Tuple[float,int,NodeInfo]] = []  # (h, counter, node)
    start = NodeInfo(START, None, None, 0, 0.0, heuristic(START))
    heapq.heappush(frontier_heap, (start.h, counter, start)); counter += 1
    nodes: Dict[int, NodeInfo] = {START: start}
    explored = set()
    step = 0
    current_newly_added = 0
    while frontier_heap:
        frontier_list = [item[2] for item in frontier_heap]
        h, _, node = heapq.heappop(frontier_heap)
        if DEBUG:
            step_yaml = f"Step {step}: frontier = {format_frontier(frontier_list)}"
            print(step_yaml)
            tree_yaml = format_tree_yaml(nodes, START)
            print(tree_yaml)
            remove_yaml = f"Remove: {cities[node.state]} (h={h})"
            print(remove_yaml)
        if DEBUG:
            added_yaml = f"Frontier Added: {current_newly_added}"
            draw_from_yaml(tree_yaml, extra_text=step_yaml + '\n' + remove_yaml + '\n' + added_yaml, filename=f'search_tree/greedy_step_{step}')
            with open('search_tree/greedy_steps.yaml', 'a', encoding='utf-8') as tf:
                tf.write(f"Greedy Step {step}:\n{tree_yaml}\n")
        if node.state == GOAL:
            print('Goal found')
            return nodes, reconstruct_path(nodes, GOAL), step
        if node.state in explored:
            step += 1
            continue
        explored.add(node.state)
        current_newly_added = 0
        for nbr in G.neighbors(node.state):
            # Don't expand neighbors already explored or already in the
            # frontier. This prevents cycles and duplicates.
            # TODO: marking duplicates here avoids cycles; reconsider if
            # you want to allow re-insertion with different heuristics.
            if nbr in explored or nodes.get(nbr) and any(n.state==nbr for _,_,n in frontier_heap):
                continue
            g = node.g + edge_cost(node.state, nbr)
            ni = NodeInfo(nbr, node.state, nbr, node.depth + 1, g, heuristic(nbr))
            nodes[nbr] = ni
            heapq.heappush(frontier_heap, (ni.h, counter, ni)); counter += 1
            current_newly_added += 1
        step += 1
    return nodes, [], step

def run_astar(debug=False):
    print('\n=== A* ===')
    counter = 0
    frontier_heap: List[Tuple[float,int,NodeInfo]] = []  # (f, counter, node)
    start = NodeInfo(START, None, None, 0, 0.0, heuristic(START))
    heapq.heappush(frontier_heap, (start.g + start.h, counter, start)); counter += 1
    nodes: Dict[int, NodeInfo] = {START: start}
    explored = set()
    step = 0
    current_newly_added = 0
    while frontier_heap:
        frontier_list = [item[2] for item in frontier_heap]
        f, _, node = heapq.heappop(frontier_heap)
        if DEBUG:
            step_yaml = f"Step {step}: frontier = {format_frontier(frontier_list)}"
            print(step_yaml)
            tree_yaml = format_tree_yaml(nodes, START)
            print(tree_yaml)
            remove_yaml = f"Remove: {cities[node.state]} (f={f}, g={node.g}, h={node.h})"
            print(remove_yaml)
        if DEBUG:
            added_yaml = f"Frontier Added: {current_newly_added}"
            draw_from_yaml(tree_yaml, extra_text=step_yaml + '\n' + remove_yaml + '\n' + added_yaml, filename=f'search_tree/astar_step_{step}')
            with open('search_tree/astar_steps.yaml', 'a', encoding='utf-8') as tf:
                tf.write(f"A* Step {step}:\n{tree_yaml}\n")
        if node.state == GOAL:
            print('Goal found')
            return nodes, reconstruct_path(nodes, GOAL), step
        if node.state in explored:
            step += 1
            continue
        explored.add(node.state)
        current_newly_added = 0
        for nbr in G.neighbors(node.state):
            new_g = node.g + edge_cost(node.state, nbr)
            new_h = heuristic(nbr)
            # If neighbor already explored with an equal-or-better g cost,
            # skip adding it. This prevents cycles and redundant higher-cost
            # paths.
            # TODO: cycle avoidance present here; consider alternative
            # handling for tie-breaking or re-opening nodes.
            if nbr in explored and nodes.get(nbr) and nodes[nbr].g <= new_g:
                continue
            if (not nodes.get(nbr)) or new_g < nodes[nbr].g:
                ni = NodeInfo(nbr, node.state, nbr, node.depth + 1, new_g, new_h)
                nodes[nbr] = ni
                heapq.heappush(frontier_heap, (ni.g + ni.h, counter, ni)); counter += 1
                current_newly_added += 1
        step += 1
    return nodes, [], step

from threading import Thread

def main(func):
    nodes, path, steps = func()
    if path:
        names = [cities[i] for i in path]
        final_cost = nodes[path[-1]].g
        print(f"Path: {' -> '.join(names)}")
        print(f"Path cost: {final_cost}")
        print(f"Expanded nodes count: {len([n for n in nodes.values() if n.parent is not None])}")
        return names, final_cost, steps
    else:
        print('No path found')
        return None, None, None

if __name__ == '__main__':
    # 创建文件夹search_tree
    import os
    os.makedirs('search_tree', exist_ok=True)
    # Run all algorithms and print summary
    # for func in (run_bfs, run_dfs, run_ucs, run_greedy, run_astar):
    #     # Thread(target=main, args=(func,)).start()
    #     main(func)
    names1, final_cost1, steps1 = main(run_bfs)
    names2, final_cost2, steps2 = main(run_dfs)
    names3, final_cost3, steps3 = main(run_ucs)
    names4, final_cost4, steps4 = main(run_greedy)
    names5, final_cost5, steps5 = main(run_astar)

    print('\nSummary of all searches:')
    print(f"BFS: Path cost = {final_cost1}, Steps = {steps1}, Path = {' -> '.join(names1) if names1 else 'No path found'}")
    print(f"DFS: Path cost = {final_cost2}, Steps = {steps2}, Path = {' -> '.join(names2) if names2 else 'No path found'}")
    print(f"UCS: Path cost = {final_cost3}, Steps = {steps3}, Path = {' -> '.join(names3) if names3 else 'No path found'}")
    print(f"Greedy: Path cost = {final_cost4}, Steps = {steps4}, Path = {' -> '.join(names4) if names4 else 'No path found'}")
    print(f"A*: Path cost = {final_cost5}, Steps = {steps5}, Path = {' -> '.join(names5) if names5 else 'No path found'}")
    
    print('\nFinished all searches.')
