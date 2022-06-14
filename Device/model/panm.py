from torch import empty, zero_
import zmq
import itertools
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

# import sys
# sys.path.append("../device")

from Device.model.topology import Topology
from Device.peer.peer import Node
from collections import defaultdict
import numpy as np
import random
import math


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
        super().__init__(self, num_nodes=n, directed=directed)

        self.k = k
        self.l = l
        self.n = n
        self.tau = tau
        self.alpha = alpha
        self.e = e
        self.T1 = T1
        self.T2 = T2

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
            stubs = list(range(n)) * d

            while stubs:
                potential_edges = defaultdict(lambda: 0)
                random.seed.shuffle(stubs)
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

    def sample_candidate_list(self, node: Node, l: int) -> Set[Node]:
        """
        This function samples $l$ nodes from non-neighbor nodes for a node $i$
        """
        non_neigh_nodes = []
        for enum_node in self.nodes:
            if enum_node not in self.adj_list[node]:
                non_neigh_nodes.append(enum_node)

        candidate_list = random.sample(non_neigh_nodes, l)
        return set(candidate_list)

    def cosine_similarity(self, node1: Node, node2: Node, alpha: float) -> float:
        """
        This function will find the similarity $s_{i,j}$ for two nodes node1 and node2
        """

        def _cos1(node1: Node, node2: Node) -> float:
            pass

        def _cos2(node1: Node, node2: Node) -> float:
            pass

        return alpha * _cos1(node1, node2) + (1 - alpha) * _cos2(node1, node2)

    def ConfigNeighInit(self, node: Node, k: int, l: int, alpha: float) -> None:
        """
        Confident Neighbour Initialization
        """
        candidate_list = self.sample_candidate_list(node, l)
        sampling_set = candidate_list.union(self.adj_list[node])

        def _compute_similarity(
            sampling_set: Dict[Node, Set[Node]], alpha: float
        ) -> Dict[Node, float]:
            dict = defaultdict(lambda: math.inf)
            for node_neigh in sampling_set:
                dict[node_neigh] = self.cosine_similarity(node, node_neigh, alpha)

            return dict

        def _sample_neigh(similarities: Dict[Node, float], k: int) -> Set[Node]:
            """
            This function solves the "Maximum sum subsequence of length k" problem using the dynamic progamming approach
            Time Complexity: O(n^2*k)
            """

            n = len(similarities)
            keys = list(similarities)
            # In the implementation dp[n][k] represents
            # maximum sum subsequence of length k and the
            # subsequence is
            # ending at index n.
            dp: TList[TList[int]] = [-1] * n
            dp_neigh: TList[TList[int]] = [set()] * n
            ans = -1

            # Initializing whole multidimensional
            # dp array with value - 1
            for i in range(n):
                dp[i] = [-1] * (k + 1)
                dp_neigh[i] = [set()] * (k + 1)

            # For each ith position increasing subsequence
            # of length 1 is equal to that array ith value
            # so initializing dp[i][1] with that array value
            for i in range(n):
                dp[i][1] = similarities[keys[i]]
                dp_neigh[i][1].add(keys[i])

            # Starting from 1st index as we have calculated
            # for 0th index. Computing optimized dp values
            # in bottom-up manner
            for i in range(1, n):
                elem = -1
                for j in range(i):

                    # check for increasing subsequence
                    if similarities[keys[j]] < similarities[keys[i]]:
                        for l in range(1, k):

                            # Proceed if value is pre calculated
                            if dp[j][l] != -1:

                                # Check for all the subsequences
                                # ending at any j < i and try including
                                # element at index i in them for
                                # some length l. Update the maximum
                                # value for every length.
                                temp = dp[i][l + 1]
                                dp[i][l + 1] = max(
                                    dp[i][l + 1], dp[j][l] + similarities[keys[i]]
                                )
                                if dp[i][l + 1] != temp:
                                    if len(dp_neigh[i][l + 1]) != 0:
                                        dp_neigh[i][l + 1].remove(elem)
                                    dp_neigh[i][l + 1].add(keys[j])
                                    elem = keys[j]

            # The final result would be the maximum
            # value of dp[i][k] for all different i.
            neigh = set()
            for i in range(n):
                if ans < dp[i][k]:
                    ans = dp[i][k]
                    neigh = dp_neigh[i][k]

            # When no subsequence of length k is
            # possible sum would be considered zero
            return neigh

        similarities = _compute_similarity(sampling_set, alpha)
        new_neigh = _sample_neigh(similarities, k)
        self.adj_list[node] = new_neigh

    def HeurNeighMatch(self, node: Node, l: int, alpha: int) -> None:
        """
        Heuristic Neighbour Matching
        """
        candidate_list = self.sample_candidate_list(node, l)
        selected_neigh_list = set(random.sample(self.adj_list[node], l))
        M: Set[Node] = candidate_list.union(selected_neigh_list)

        def _compute_similarity(M: Dict[Node, Set[Node]]) -> Dict[Node, float]:
            dict = defaultdict(lambda: math.inf)
            for node_neigh in M:
                dict[node_neigh] = self.cosine_similarity(node, node_neigh, alpha)

            return dict

        similarities = _compute_similarity(M)

        def EM_init(sampling_list: TList[Node]) -> TList[TList[int]]:
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

        ## EM-Optimization step

        # 1) Initialization
        sampling_list = list(M)
        gamma = EM_init(sampling_list)
        gamma = np.array(gamma)
        gamma_next = np.zeros(gamma.shape)
        similarities = np.array(similarities)

        assert gamma.shape[1] == len(sampling_list) == similarities.shape[0]

        while gamma_next != gamma:

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
        for j in len(sampling_list):
            if gamma[0][j] == 1:
                H.add(sampling_list[j])

        self.adj_list[node] = H.union(
            self.adj_list[node].difference(selected_neigh_list)
        )

    def train(self):

        for round in range(1, self.T1 + self.T2):
            for node in self.nodes:

                node.training_step(round, self.e)
                if round < self.T1:
                    if round != 1:
                        self.ConfigNeighInit(node, k=self.k, alpha=self.alpha, l=self.l)
                    node.Aggregation()
                else:
                    if round % self.tau == 0:
                        self.HeurNeighMatch(node, l=self.l, alpha=self.alpha)
                        node.GossipAggre()
                    else:
                        node.GossipAggre()
