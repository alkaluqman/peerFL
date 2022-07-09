import os, time
import zmq
import zmq_helper
import training, inference
import json, joblib
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

class Node:
    """A peer-to-peer node that can act as client or server at each round"""

    def __init__(self, context, node_id, node_type, server_id, num_clients):
        self.context = context
        self.node_id = node_id
        self.log_prefix = "[" + str(node_id).upper() + "] "
        self.node_type = node_type  # either server or client specific to one round
        self.server_id = server_id
        self.total_clients = num_clients
        self.socket = self.initialize_node()
        self.local_model = None  # local model

    def initialize_node(self):
        if self.node_type == 'server':
            return zmq_helper.act_as_server(self.context, self.node_id)
        elif self.node_type == 'client':
            return zmq_helper.act_as_client(self.context, self.server_id, self.node_id)
        else:
            print("Incorrect node type")
        return None

    def print_node_details(self):
        print("*" * 60)
        print("%s Node ID = %s" % (self.log_prefix, self.node_id))
        print("%s Node Type = %s" % (self.log_prefix, self.node_type))
        print("%s Server ID = %s" % (self.log_prefix, self.server_id))
        print("%s Context = %s" % (self.log_prefix, self.context))
        print("%s Socket = %s" % (self.log_prefix, self.socket))
        print("%s Model = %s" % (self.log_prefix, self.local_model))
        print("*" * 60)

    def save_model(self):
        model_filename = "/usr/thisdocker/dataset/" + str(self.node_id) + ".pkl"
        joblib.dump(self.local_model, model_filename)

    def training_step(self, step):
        # local model training
        self.local_model = training.local_training(self.node_id)
        # self.local_model = {"from": self.node_id}  # for debugging
        # self.save_model()

        print("%sSending iteration %s …" % (self.log_prefix, str(step)))
        zmq_helper.send_zipped_pickle(self.socket, self.local_model)

        # wait for global model
        self.local_model = zmq_helper.recv_zipped_pickle(self.socket)
        print("%sReceived object %s version %s" % (self.log_prefix, str(self.local_model), str(step)))
        self.save_model()

    def inference_step(self):
        models_received_from_clients = 0
        client_list = []
        client_dict = {}
        while True:
            #  Wait for next request from client
            client_identity = self.socket.recv(0)  # Reads identity
            client_model = zmq_helper.recv_zipped_pickle(self.socket)  # Reads model object
            print("%sReceived object from %s" % (self.log_prefix, client_identity.decode("utf-8")))
            # print(client_identity.decode("utf-8"), "***", client_model)
            client_list.append(client_identity)
            client_dict[client_identity] = client_model
            models_received_from_clients += 1  # TBD: replace with check if model received per identity using dictionary
            if models_received_from_clients == self.total_clients -1:
                # FedAvg to get global model
                self.local_model = inference.FedAvg(client_dict)
                # self.local_model = {"updated":"model"}  # for debugging

                # Sending global model
                for i in client_list:
                    print("%sSending global model object to %s" % (self.log_prefix, i.decode("utf-8")))
                    # socket.send_multipart([i, b"yoooooo"])
                    zmq_helper.send_updated_model(self.socket, self.local_model, i)
                return  # reset for next epoch


def main():
    """main function"""
    context = zmq.Context()  # We should only have 1 context which creates any number of sockets
    node_id = os.environ["ORIGIN"]
    num_clients = int(os.environ["NUM_CLIENTS"])

    # Read comm template config file
    comm_template = json.load(open('comm_template.json'))
    total_rounds = len(comm_template.keys())

    for i in range(1,total_rounds+1):
        ith_round = comm_template[str(i)]
        server = "node" + str(ith_round["server"])
        clients = ith_round["clients"]
        clients = ["node" + str(c) for c in clients]
        if node_id == server :
            this_node = Node(context, node_id, "server", server, num_clients)
            inference_start = time.process_time()
            this_node.inference_step()
            print("%sTime : Inference Step = %s" % (this_node.log_prefix, str(time.process_time() - inference_start)))
        elif node_id in clients :
            this_node = Node(context, node_id, "client", server, num_clients)
            training_start = time.process_time()
            this_node.training_step(i)
            print("%sTime : Training Step = %s" % (this_node.log_prefix, str(time.process_time()-training_start)))
        else :
            print("Error initializing node %s in round %s" % (node_id, i))
        # this_node.print_node_details()


if __name__ == "__main__":
    main()
