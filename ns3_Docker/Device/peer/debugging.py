import os, time
import random

# import zmq
# import zmq_helper
import ns_helper_new
import json, joblib
import ast
# import training, inference

class Node:
    """A peer-to-peer node that can act as client or server at each round"""

    def __init__(self, context, node_id, peers):
        self.context = context
        self.node_id = node_id
        self.peers = peers
        self.log_prefix = "[" + str(node_id).upper() + "] "
        self.in_connection = None
        self.out_connection = {}
        self.local_model = None  # local model
        self.local_history = None
        self.ns_helper = ns_helper_new.NsHelper()
        self.initialize_node()

    def initialize_node(self):
        """Creates one zmq.ROUTER socket as incoming connection and n number
        of zmq.DEALER sockets as outgoing connections as per adjacency matrix"""
        self.local_history = []
        # self.in_connection = zmq_helper.act_as_server(self.context, self.node_id)
        # if len(self.peers) > 0:
        #     for server_id in self.peers :
        #         self.out_connection[server_id] = zmq_helper.act_as_client(self.context, server_id, self.node_id)

        numNodes = 10
        self.ns3Nodes = self.ns_helper.createNodes(numNodes)
        self.ns3Interface = self.ns_helper.createInterface(self.ns3Nodes)
        server_num = int(self.node_id[-1])
        
        for peer in self.peers:
            client_num = int(peer[-1])
            self.in_connection = self.ns_helper.act_as_client(self.ns3Nodes, client_num)
            self.out_connection[client_num] = self.ns_helper.act_as_server(self.ns3Nodes, self.ns3Interface, server_num, client_num)

        #self.in_connection = self.ns_helper.act_as_client(self.ns3Nodes, node_num)
        #if len(self.peers) > 0:
        #    for server_id in self.peers :
        #        server_num = int(server_id[-1])
        #        self.out_connection[server_id] = self.ns_helper.act_as_server(self.ns3Nodes, self.ns3Interface, server_num, node_num)


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
            # zmq_helper.send_zipped_pickle(self.out_connection[to_node], self.local_model)
            # self.out_connection[to_node].send_string(self.local_model)
            self.ns_helper.send_data(self.out_connection[int(to_node[-1])], self.local_model, 0.0)
            self.ns_helper.simulation_start()
        except Exception as e:
            print("%sERROR establishing socket : To-node not in peers" % self.log_prefix)

    def receive_model(self):
        # from_node = self.in_connection.recv(0)  # Reads identity
        # self.local_model = zmq_helper.recv_zipped_pickle(self.in_connection)  # Reads model object
        # self.local_model = self.in_connection.recv_string()
        # return from_node

        received = self.ns_helper.recv_data(self.in_connection)
        self.ns_helper.simulation_start()  # Force order of execution

        received = self.ns_helper.read_recvd_data(0)
        self.ns_helper.simulation_start()
        return received

    def training_step(self, step):
        # local model training
        # build_flag = True if step == 1 else False
        # self.local_model = training.local_training(self.node_id, self.local_model, build_flag)
        #time.sleep(1)
        self.local_model = {"from": self.node_id}  # for debugging
        # self.save_model()

    # def inference_step(self):
    #     inference.eval_on_test_set(self.local_model)

def main():
    """main function"""
    random.seed(42)
    context = None # zmq.Context()  # We should only have 1 context which creates any number of sockets
    node_id = os.environ["ORIGIN"]
    print(node_id)
    peers_list = ast.literal_eval(os.environ["PEERS"])
    print(peers_list)
    this_node = Node(context, node_id, peers_list)

    # Read comm template config file
    comm_template = json.load(open('comm_template.json'))
    #total_rounds = len(comm_template.keys())  # one entry is for transfer
    epochs = 1

    for e in range(1, epochs+1):
        from_node = "node" + str(1)
        to_node = "node" + str(2)
        if node_id == from_node:
            # This node is dealer and receiving node is router
            training_start = time.process_time()
            this_node.training_step(None)
            print("%sTime : Training Step = %s" % (this_node.log_prefix, str(time.process_time() - training_start)))

            print("%sSending iteration %s from %s to %s" % (this_node.log_prefix, str(1), from_node, to_node))
            this_node.send_model(to_node)
        elif node_id == to_node:
            # This node is router and sending node is dealer
            rcvd_from = this_node.receive_model()
            i = 0
            while(not rcvd_from and i <= 20):
                rcvd_from = this_node.receive_model()
                i+=1
            print("%sReceived object %s at iteration %s" % (this_node.log_prefix, str(this_node.local_model), str(1)))

if __name__ == "__main__":
    main()


#sender_id = "node1"
#receiver_id = "node2"
#sender_peer_list = ["node2"]
#sender_Node = Node(None, sender_id, sender_peer_list)
#sender_Node.training_step(None)
#sender_Node.send_model(2)
#sender_Node.receive_model()
#print("%sReceived object %s at iteration %s" % (sender_Node.log_prefix, str(sender_Node.local_model), str(1)))


