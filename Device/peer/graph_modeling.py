from collections import defaultdict
import random
import sys
from copy import deepcopy
from queue import PriorityQueue
import math
from py2opt.routefinder import RouteFinder # pip install py2opt


random.seed(1421)
# inf = float('inf')
inf = sys.maxsize
# inf = math.inf


def add_node(graph, n, nodeScore, nodes, nodes_no):
    """Add a node to the set of nodes and the graph"""
    if n in nodes.keys():
        print("Node", n, "already exists")
    else:
        nodes_no = nodes_no + 1
        nodes[n] = nodeScore
        if nodes_no > 1:
            for node in graph:
                node.append(0)
        temp = []
        for i in range(nodes_no):
            temp.append(0)
        graph.append(temp)

        return graph, nodes, nodes_no


def add_edge(graph, nodes_no, nodes, n1, n2, e):
    """Add an edge between node n1 and n2 with edge weight e"""
    # Check if node n1 is a valid node
    if n1 not in nodes:
        print("Node", n1, "does not exist.")
    # Check if node n1 is a valid node
    elif n2 not in nodes:
        print("Node", n2, " does not exist.")
    # Since this code is not restricted to a directed or
    # an undirected graph, an edge between n1 n2 does not
    # imply that an edge exists between n2 and n1
    else:
        graph[n1][n2] = e

    return graph, nodes, nodes_no


def make_graph(num_clients, dummy_clients, rando=1421):
    random.seed(rando)
    # Driver code
    # stores the nodes in the graph
    nodes = defaultdict(list)
    # stores the number of nodes in the graph
    nodes_no = 0
    graph = []

    # Add nodes to the graph
    for i in range(num_clients + dummy_clients):
        graph, nodes, nodes_no = add_node(graph, i, random.randint(-5, 5), nodes, nodes_no)

    # Add the edges between the nodes by specifying
    # the from and to node along with the edge weights.

    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 1, dummy_clients - 1 + 3,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 1, dummy_clients - 1 + 6,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 2, dummy_clients - 1 + 6,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 3, dummy_clients - 1 + 7,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 4, dummy_clients - 1 + 1,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 4, dummy_clients - 1 + 5,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 5, dummy_clients - 1 + 2,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 6, dummy_clients - 1 + 4,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 6, dummy_clients - 1 + 5,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 7, dummy_clients - 1 + 2,
                                      random.randint(0, 10))
    graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, dummy_clients - 1 + 7, dummy_clients - 1 + 1,
                                      random.randint(0, 10))

    # if (dummy_clients>0):
    #     graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, 0, 1, 0)
    for i in range(dummy_clients):
        for j in range(dummy_clients, num_clients + dummy_clients):
            graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, i, j, 0)
            graph, nodes, nodes_no = add_edge(graph, nodes_no, nodes, j, i, 0)

    for srcNode in range(nodes_no):
        for dstNode in range(nodes_no):
            if (srcNode != dstNode and srcNode < dummy_clients and dstNode < dummy_clients):
                graph[srcNode][dstNode] = inf
            if (srcNode == dstNode or srcNode < dummy_clients or dstNode < dummy_clients):
                continue
            tmp = graph[srcNode][dstNode]
            if (tmp == 0):
                graph[srcNode][dstNode] = inf
    return graph, nodes_no, nodes

def print_graph(graph, nodes_no, nodes):
    """Print the graph (adjacency matrix) with source, dest, edge weight and node scores"""
    for srcNode in range(nodes_no):
    #print()
    for dstNode in range(nodes_no):
        edwt = graph[srcNode][dstNode]
        if (edwt==0 or edwt>=inf):
            continue
        print(f'{srcNode}({nodes[srcNode]}) -> {dstNode}({nodes[dstNode]}) \t| Edge weight: {edwt}')

def print_graph2(graph):
    """Print graph as an adjacency matrix"""
    print('\n'.join([''.join(['{:4}'.format(item if item<inf else float('inf')) for item in row]) for row in graph]))

def printGraph(G):
    for row in G:
        print(row)

def FloydWarshall(G, show=0):
    assert len(G) == len(G[0])
    M = deepcopy(G)
    n = len(M[0])
    routeG = [['SELF'] * n for i in range(n)]
    nxt = [[0] * n for i in range(n)]

    for i in range(n):
        for j in range(n):
            if (G[i][j] < inf):
                nxt[i][j] = j
    for k in range(n):
        if show:
            print('k=%d' % (k - 1))
            printGraph(M)
            print('-' * 10)
        for i in range(n):
            for j in range(n):
                tmpsum = M[i][k] + M[k][j]
                if (M[i][j] > tmpsum):
                    M[i][j] = tmpsum
                    nxt[i][j] = nxt[i][k]

    print("APSP (k=%d):" % k)
    print("All pair shortest distance:")
    printGraph(M)

    for i in range(n):
        for j in range(n):
            if (i != j):
                path = [i]
                while path[-1] != j:
                    path.append(nxt[path[-1]][j])
                routeG[i][j] = path

    return M, routeG

num_clients = 7
dummy_clients = 0
graph, nodes_no, nodes = make_graph(num_clients, dummy_clients, 73)
# print("Adjacency matrix of base graph:")
# print_graph2(graph)
# print()

# Get route graph
newG, route = FloydWarshall(graph, 0)
# print()
# print("Route graph for each pair:")
# printGraph(route)

# Add dummy to avoid deciding origin:
newG.insert(0, [0 for i in range(num_clients+1)])
for i in range(1, num_clients+1):
    # newG[i].insert(0, inf)
    newG[i].insert(0, 0)
# print_graph2(newG)

# 2 OPT Heuristic with TSP
dummy = 'D_0'
num_iter = 20
fin_paths = 5

top_k_paths = PriorityQueue(fin_paths+1)
all_paths = []
top_k_paths_list = []


# Set client names for the py2opt solver
client_names = [dummy]
for i in range(num_clients):
    client_names.append(str(i+1))

# Find top k paths from a bunch of possible paths
for i in range(num_iter):
    route_finder = RouteFinder(newG, client_names, iterations=1)
    best_distance, best_route = route_finder.solve()
    print("best distance:", best_distance)
    # print("best shortest route:", best_route)
    if best_route in all_paths:  # To ensure all paths are unique
        continue
    all_paths.append(best_route)
    top_k_paths.put((-1 * best_distance, best_route))  # -1 factor because we want minimized values
    if (top_k_paths.full()):  # Remove low priority paths
        top_k_paths.get()

# Store the top k paths in a list for easier access
# Multiply with -1 again to have normal values back
print("\nNew Graph path:")
for j in range(top_k_paths.qsize()):
    dist, pth = top_k_paths.get()
    top_k_paths_list.append((-1 * dist, pth))
    print(top_k_paths_list[-1])

# Get actual paths in og graph from the sub-optimal path in new graph that I created
final_k_paths = deepcopy(top_k_paths_list)
print("\nActual path:")
for i in range(len(final_k_paths)):
    tempPath = [int(final_k_paths[i][1][1])] # 0th element is D_0
    for j in range(1, len(final_k_paths[i][1])-1):
        src = int(final_k_paths[i][1][j])
        dst = int(final_k_paths[i][1][j+1])
        for k in route[src-1][dst-1]:
            if (int(k)==src-1):
                continue
            tempPath.append(int(k+1))
    final_k_paths[i] = list(final_k_paths[i])
    final_k_paths[i][1] = tempPath
    final_k_paths[i] = tuple(final_k_paths[i])
    print(final_k_paths[i])

print()
print("Actual 0 indexed path:")
final_0idx_k_path = []
for rows in final_k_paths:
    ty = (rows[0], [x-1 for x in rows[1]])
    final_0idx_k_path.append(ty)
    print(final_0idx_k_path[-1])

print()
## To verify my algo gives right scores by the resultant route/path as intended
for idx, (best_distance,idx_0_path) in enumerate(final_0idx_k_path):
    tmp=0
    for i in range(len(idx_0_path)-1):
        tmp+=graph[idx_0_path[i]][idx_0_path[i+1]]
    # print("Verification sum:", tmp)
    print(f'Is the distance for path number {idx+1} verified?', "YES! :)" if tmp==best_distance else "NO :(")

print(final_0idx_k_path)
print(final_k_paths)
print("Order = ",final_k_paths[-1][1])
print("Cost = ",final_k_paths[-1][0])