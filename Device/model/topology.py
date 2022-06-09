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

from Device.peer.peer import Node
import zmq


class Topology:
    def __init__(self, num_nodes: int, directed: bool = False) -> None:
        self.num_nodes = num_nodes
        self.directed = directed
        self.nodes = [Node(zmq.context(), i, []) for i in range(self.num_nodes)]
        self.adj_list = {node: set() for node in self.nodes}

    def add_edge(self, node1: Node, node2: Node, weight: int = 0) -> Node:

        self.adj_list[node1].add((node2, weight))

        if not self.directed:
            self.adj_list[node2].add((node1, weight))

    def print_topology(self):
        for key in self.adj_list.keys():
            return (key, ": ", self.adj_list[key])

    @abstractmethod
    def initNeighList(self):
        pass
