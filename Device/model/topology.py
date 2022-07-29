from abc import abstractmethod
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List as TList,
    Any,
    Optional,
    Set,
    Tuple,
    Type as PythonType,
)


class GNode:
    def __init__(self, node_id: int, peers: TList["GNode"]):
        self.node_id = node_id
        self.peers = peers

    def __str__(self):
        return "node{}".format(self.node_id)

    def __repr__(self):
        return self.__str__()


class Topology:
    def __init__(self, num_nodes: int, directed: bool = False) -> None:
        self.num_nodes = num_nodes
        self.directed = directed
        self.nodes = [GNode(i, []) for i in range(1, self.num_nodes + 1)]
        self.adj_list = {node: set() for node in self.nodes}

    def add_edge(self, node1: GNode, node2: GNode, weight: int = 0) -> None:

        self.adj_list[node1].add((node2, weight))

        if not self.directed:
            self.adj_list[node2].add((node1, weight))

    def print_topology(self):
        for key in self.adj_list.keys():
            return (key, ": ", self.adj_list[key])

    @abstractmethod
    def initNeighList(self):
        pass
