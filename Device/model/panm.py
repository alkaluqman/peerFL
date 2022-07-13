import zmq
import os
import numpy as np
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

import sys
sys.path.append('/workspace/')
sys.path.append('/workspace/peer')

from model.topology import Topology, GNode
from peer.peer import Node
from collections import defaultdict
import numpy as np
import random
import math


def list_to_np():
    pass


class PANM(Topology):
    def __init__(
        self,
        n: int,
        k: int,
        l: int,
        e: int,
        tau: int,
        alpha: float,
        T1: int,
        T2: int,
        directed: bool = False,
    ) -> None:
        """
        n : Number of all clients
        k : Size of aggregation neighbours
        l : Size of neighbour candidate list
        tau : Round interval of HNM in stage two
        alpha : Hyperparmeter in gradient-based metric
        """
        super().__init__(num_nodes=n, directed=directed)

        self.k = k
        self.l = l
        self.n = n
        self.tau = tau
        self.alpha = alpha
        self.e = e
        self.T1 = T1
        self.T2 = T2
        self.initNeighList()

    def initNeighList(self) -> None:
        """
        This function generates a random $k$-regular undirected graph on $n$ nodes
        """

        if (self.n * self.k) % 2 != 0:
            raise RuntimeError("n * k must be even")
        if not 0 <= self.k < self.n:
            raise RuntimeError("the 0 <= k < n inequality must be satisfied")
        if self.k == 0:
            return

        def _suitable(edges, potential_edges):
            # Helper subroutine to check if there are suitable edges remaining
            # If False, the generation of the graph has failed
            if not potential_edges:
                return True
            for s1 in potential_edges:
                for s2 in potential_edges:
                    # Two iterators on the same dictionary are guaranteed
                    # to visit it in the same order if there are no
                    # intervening modifications.
                    if s1 == s2:
                        # Only need to consider s1-s2 pair one time
                        break
                    if s1 > s2:
                        s1, s2 = s2, s1
                    if (s1, s2) not in edges:
                        return True
            return False

        def _try_creation():
            # Attempt to create an edge set

            edges = set()
            stubs = list(range(self.n)) * self.k

            while stubs:
                potential_edges = defaultdict(lambda: 0)
                random.shuffle(stubs)
                stubiter = iter(stubs)
                for s1, s2 in zip(stubiter, stubiter):
                    if s1 > s2:
                        s1, s2 = s2, s1
                    if s1 != s2 and ((s1, s2) not in edges):
                        edges.add((s1, s2))
                    else:
                        potential_edges[s1] += 1
                        potential_edges[s2] += 1

                if not _suitable(edges, potential_edges):
                    return None  # failed to find suitable edge set

                stubs = [
                    node
                    for node, potential in potential_edges.items()
                    for _ in range(potential)
                ]
                return edges

        # Even though a suitable edge set exists,
        # the generation of such a set is not guaranteed.
        # Try repeatedly to find one.
        edges = _try_creation()
        while edges is None:
            edges = _try_creation()

        for edge in edges:
            self.add_edge(self.nodes[edge[0]], self.nodes[edge[1]])
            self.nodes[edge[0]].peers.append(self.nodes[edge[1]])
            self.nodes[edge[1]].peers.append(self.nodes[edge[0]])

    def sample_candidate_list(self, node: GNode, l: int) -> Set[Node]:
        """
        This function samples $l$ nodes from non-neighbor nodes for a node $i$
        """
        non_neigh_nodes = []
        for enum_node in self.nodes:
            if (enum_node, 0) not in self.adj_list[node]:
                non_neigh_nodes.append(enum_node)

        candidate_list = random.sample(non_neigh_nodes, l)
        return set(candidate_list)

    def cosine_similarity(self, node1: Node, node2: GNode, alpha: float) -> float:
        """
        This function will find the similarity $s_{i,j}$ for two nodes node1 and node2
        """

        def _cos1(node1: Node, node2: GNode) -> float:

            w1 = np.subtract(
                np.array(node1.local_model.get_weights(), dtype=object),
                np.array(node1.prev_local_model.get_weights(), dtype=object),
            )
            w2 = np.subtract(
                np.array(node1.peer_models[str(node2)].get_weights(), dtype=object),
                np.array(
                    node1.prev_peer_models[str(node2)].get_weights(), dtype=object
                ),
            )
            # vec1 = w1.flatten()
            # vec2 = w2.flatten()
            # unit_vec1 = vec1/np.linalg.norm(vec1)
            # unit_vec2 = vec2/np.linalg.norm(vec2)

            # return np.dot(unit_vec1, unit_vec2)
            return random.uniform(0, 1)

        def _cos2(node1: Node, node2: GNode) -> float:
            w1 = np.subtract(
                np.array(node1.local_model.get_weights(), dtype=object),
                np.array(node1.initial_model.get_weights(), dtype=object),
            )
            w2 = np.subtract(
                np.array(node1.peer_models[str(node2)].get_weights(), dtype=object),
                np.array(node1.initial_model.get_weights(), dtype=object),
            )
            # vec1 = w1.flatten()
            # vec2 = w2.flatten()
            # unit_vec1 = vec1/np.linalg.norm(vec1)
            # unit_vec2 = vec2/np.linalg.norm(vec2)

            # return np.dot(unit_vec1, unit_vec2)
            return random.uniform(0, 1)

        return alpha * _cos1(node1, node2) + (1 - alpha) * _cos2(node1, node2)

    def ConfigNeighInit(self, this_node: Node, k: int, l: int, alpha: float) -> None:
        """
        Confident Neighbour Initialization
        """
        node = self.nodes[int(this_node.node_id[-1]) - 1]
        candidate_list = self.sample_candidate_list(node, l)
        sampling_set = candidate_list.union(set([i[0] for i in self.adj_list[node]]))

        def _compute_similarity(
            sampling_set: Dict[GNode, Set[GNode]], alpha: float
        ) -> Dict[GNode, float]:
            dict = defaultdict(lambda: math.inf)
            for node_neigh in sampling_set:
                dict[node_neigh] = self.cosine_similarity(this_node, node_neigh, alpha)

            return dict

        def _sample_neigh2(similarities: Dict[GNode, float], k: int) -> Set[GNode]:
            """
            This function solves the "Maximum sum subsequence of length k" problem using priority queue
            """
            n = len(similarities)
            nums = list(similarities.values())
            keys = list(similarities.keys())

            sortedArray = nums.copy()
            sortedArray.sort()

            d = defaultdict(lambda: 0)
            neigh = [None] * k

            for i in range(len(sortedArray) - k, len(sortedArray)):
                d[sortedArray[i]] = d.get(sortedArray[i], 0) + 1

            j = 0

            for n in nums:
                if j < k and n in d:
                    neigh[j] = keys[nums.index(n)]
                    if d[n] == 1:
                        d.pop(n)
                    elif d[n] > 1:
                        d[n] -= 1
                    j += 1
            return neigh

        similarities = _compute_similarity(sampling_set, alpha)
        new_neigh = _sample_neigh2(similarities, k)
        self.adj_list[node] = set([(i, 0) for i in iter(new_neigh)])

    def HeurNeighMatch(self, this_node: Node, l: int, alpha: int) -> None:
        """
        Heuristic Neighbour Matching
        """
        node = self.nodes[int(this_node.node_id[-1]) - 1]
        candidate_list = self.sample_candidate_list(node, l)
        selected_neigh_list = set(
            random.sample(set([i[0] for i in self.adj_list[node]]), l)
        )
        M: Set[GNode] = candidate_list.union(selected_neigh_list)

        def _compute_similarity(M: Dict[GNode, Set[GNode]]) -> Dict[GNode, float]:
            dict = defaultdict(lambda: math.inf)
            for node_neigh in M:
                dict[node_neigh] = self.cosine_similarity(this_node, node_neigh, alpha)

            return dict

        similarities = _compute_similarity(M)

        def EM_init(sampling_list: TList[GNode]) -> TList[TList[int]]:
            gamma: TList[TList[int]] = [[], []]
            for node_neigh in sampling_list:
                if node_neigh in candidate_list:
                    gamma[0].append(1)
                    gamma[1].append(0)
                elif node_neigh in selected_neigh_list:
                    gamma[0].append(0)
                    gamma[1].append(1)
                else:
                    gamma[0].append(0)
                    gamma[1].append(0)

            return gamma

        ## EM-Optimization step

        # 1) Initialization
        sampling_list = list(M)
        gamma = EM_init(sampling_list)
        gamma = np.array(gamma)
        gamma_next = np.zeros(gamma.shape)
        similarities = np.array(list(similarities.items()))

        assert gamma.shape[1] == len(sampling_list) == similarities.shape[0]

        while gamma_next.all() != gamma.all():

            # 2) E-Step
            n = np.sum(gamma, axis=1)
            mu = np.dot(gamma, similarities) / n
            beta = n / len(sampling_list)
            simga = np.dot(gamma, (gamma.T - mu).T ** 2) / n

            # 3) M-Step
            gamma_next = gamma
            y = np.expand_dims(np.argmax(gamma, axis=0), axis=1)
            a = np.zeros(gamma.T.shape)
            np.put_along_axis(a, y, 1, axis=1)
            gamma = a.T

        H = set()
        for j in range(len(sampling_list)):
            if gamma[0][j] == 1:
                H.add((sampling_list[j], 0))

        selected_neigh_list_weight = set([(i, 0) for i in iter(selected_neigh_list)])
        self.adj_list[node] = H.union(
            self.adj_list[node].difference(selected_neigh_list_weight)
        )

    def send_models(self, this_node: Node):
        for neigh in self.nodes:
            if neigh == self.nodes[int(this_node.node_id[-1]) - 1]:
                continue
            this_node.send_model("node" + str(neigh.node_id))

    def train(self):

        context = (
            zmq.Context()
        )  # We should only have 1 context which creates any number of sockets
        node_id = os.environ["ORIGIN"]
        peers_list = ["node" + str(peer.node_id) for peer in self.nodes]
        this_node = Node(context, node_id, peers_list)

        for round in range(1, self.T1 + self.T2):

            this_node.training_step(round, self.e)

            if round < self.T1:
                if round != 1:
                    self.ConfigNeighInit(
                        this_node, k=self.k, alpha=self.alpha, l=self.l
                    )
                self.send_models(this_node)
                this_node.Aggregation(
                    self.adj_list[self.nodes[int(this_node.node_id[-1]) - 1]]
                )
            else:
                if round % self.tau == 0:
                    self.HeurNeighMatch(this_node, l=self.l, alpha=self.alpha)
                    self.send_models(this_node)
                    this_node.GossipAggre(
                        self.adj_list[self.nodes[int(this_node.node_id[-1]) - 1]]
                    )
                else:
                    self.send_models(this_node)
                    this_node.GossipAggre(
                        self.adj_list[self.nodes[int(this_node.node_id[-1]) - 1]]
                    )


if __name__ == "__main__":

    num_clients = os.environ["NUM_CLIENTS"]
    panm = PANM(n=num_clients, k=2, l=1, e=1, tau=2, alpha=0.5, T1=5, T2=5)
    panm.train()
