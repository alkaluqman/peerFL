from email.policy import default
import os, time
import zmq
import zmq_helper
import json, joblib
import ast
import training, inference
from collections import defaultdict
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


class Node:
    """A peer-to-peer node that can act as client or server at each round"""

    def __init__(self, context, node_id, peers):
        self.context = context
        self.node_id = node_id
        self.peers = peers
        self.log_prefix = "[" + str(node_id).upper() + "] "
        self.in_connection = None
        self.peer_models = {}
        self.prev_peer_models = {}
        self.out_connection = {}
        self.local_model = None  # local model
        self.initial_model = None
        self.prev_local_model = None
        self.local_history = None
        self.initialize_node()

    def initialize_node(self):
        """Creates one zmq.ROUTER socket as incoming connection and n number
        of zmq.DEALER sockets as outgoing connections as per adjacency matrix"""
        self.local_history = []
        self.in_connection = zmq_helper.act_as_server(self.context, self.node_id)
        if len(self.peers) > 0:
            for server_id in self.peers:
                # print(server_id)
                self.out_connection[server_id] = zmq_helper.act_as_client(
                    self.context, server_id, self.node_id
                )

    def print_node_details(self):
        print("*" * 60)
        print("%s Node ID = %s" % (self.log_prefix, self.node_id))
        print("%s Peer IDs = %s %s" % (self.log_prefix, type(self.peers), self.peers))
        print("%s Context = %s" % (self.log_prefix, self.context))
        print("%s Incoming Connection = %s" % (self.log_prefix, self.in_connection))
        print("%s Outgoing Connections = %s" % (self.log_prefix, self.out_connection))
        print("%s History = %s" % (self.log_prefix, self.local_history))
        print("*" * 60)

    def save_model(self):
        model_filename = "/usr/thisdocker/dataset/" + str(self.node_id) + ".pkl"
        joblib.dump(self.local_model, model_filename)

    def load_prev_model(self):
        model_filename = "/usr/thisdocker/dataset/" + str(self.node_id) + ".pkl"
        self.local_model = joblib.load(model_filename)

    def send_model(self, to_node):
        try:
            zmq_helper.send_zipped_pickle(
                self.out_connection[to_node], self.local_model
            )
            # self.out_connection[to_node].send_string(self.local_model)
        except Exception as e:
            print("%sERROR establishing socket for to-node" % self.log_prefix)

    def receive_model(self):
        from_node = self.in_connection.recv(0)  # Reads identity
        self.local_model = zmq_helper.recv_zipped_pickle(
            self.in_connection
        )  # Reads model object
        # self.local_model = self.in_connection.recv_string()
        return from_node

    def training_step(self, step, num_epoch=1):
        # local model training
        build_flag = True if step == 1 else False
        self.local_model, self.prev_local_model = training.local_training(
            self.node_id, self.local_model, build_flag, num_epoch
        )
        if build_flag:
            self.initial_model = self.prev_local_model
            self.peer_models = defaultdict(lambda: self.initial_model)

    def inference_step(self):
        inference.eval_on_test_set(self.local_model)

    def Aggregation(self, clients):

        models_received_from_clients = 0
        models_received_from_peers = 0
        client_list = []
        client_dict = {}
        while True:
            #  Wait for next request from client
            client_identity = self.in_connection.recv(0)  # Reads identity
            client_model = zmq_helper.recv_zipped_pickle(
                self.in_connection
            )  # Reads model object
            client_index = client_identity.decode("utf-8")
            print("%sReceived object from %s" % (self.log_prefix, client_index))
            # print(client_identity.decode("utf-8"), "***", client_model)
            # print(type(client_index))
            client_list.append(client_identity)
            client_dict[client_index] = client_model
            self.prev_peer_models[client_index] = self.peer_models[client_index]
            self.peer_models[client_index] = client_model

            if client_index in clients:
                models_received_from_clients += 1  # TBD: replace with check if model received per identity using dictionary
            models_received_from_peers += 1

            if models_received_from_clients == len(clients) - 1:
                # FedAvg to get global model
                self.local_model = inference.FedAvg(client_dict)

            if models_received_from_peers == len(self.peers) - 1:
                return

    def GossipAggre(self, clients):
        self.Aggregation(clients)


def main():
    """main function"""
    context = (
        zmq.Context()
    )  # We should only have 1 context which creates any number of sockets
    node_id = os.environ["ORIGIN"]
    peers_list = ast.literal_eval(os.environ["PEERS"])
    this_node = Node(context, node_id, peers_list)

    # Read comm template config file
    comm_template = json.load(open("comm_template.json"))
    total_rounds = len(comm_template.keys())

    for i in range(1, total_rounds + 2):
        if i == total_rounds + 1:
            if this_node.node_id == "node" + str(
                comm_template[str(total_rounds)]["to"]
            ):
                # Last node training
                this_node.training_step(i)
                # Global accuracy
                this_node.inference_step()
        else:
            ith_round = comm_template[str(i)]
            from_node = "node" + str(ith_round["from"])
            to_node = "node" + str(ith_round["to"])
            if node_id == from_node:
                # This node is dealer and receiving node is router
                training_start = time.process_time()
                this_node.training_step(i)
                print(
                    "%sTime : Training Step = %s"
                    % (this_node.log_prefix, str(time.process_time() - training_start))
                )

                print(
                    "%sSending iteration %s from %s to %s"
                    % (this_node.log_prefix, str(i), from_node, to_node)
                )
                this_node.send_model(to_node)
            elif node_id == to_node:
                # This node is router and sending node is dealer
                rcvd_from = this_node.receive_model()
                print(
                    "%sReceived object %s at iteration %s"
                    % (this_node.log_prefix, str(this_node.local_model), str(i))
                )

                this_node.save_model()

                # Logging iteration and prev_node for audit
                this_node.local_history.append(
                    {"iteration": i, "prev_node": rcvd_from.decode("utf-8")}
                )


if __name__ == "__main__":
    main()
