#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ns_helper_new
import numpy as np
import sys
import pickle
import time
#import tensorflow as tf


def create_all_connections(numNodes, ns3Nodes, ns3Interface):
    socket_dict = {}
    for i in range(numNodes):  # ns3 is zero based whereas graph is 1 based indexing
        in_socket = ns_helper.act_as_client(ns3Nodes, i)
        socket_dict[(i+1,i+1)] = in_socket
        for j in range(numNodes):
            if i!=j:
                out_socket = ns_helper.act_as_server(ns3Nodes, ns3Interface, i, j)
                socket_dict[(i+1,j+1)] = out_socket
    return socket_dict


def oneround(data, all_sockets, from_node, to_node, attime=0.0):
    # oneround("hello", all_sockets[(1, 2)], all_sockets[(2, 2)], sim_time)
    print("Sending "+str(data)+" from node"+str(from_node)+" to node"+str(to_node))
    from_socket = all_sockets[(from_node, to_node)]
    to_socket = all_sockets[(to_node, to_node)]

    ns_helper.send_data(from_socket, data, attime)

    received = ns_helper.recv_data(to_socket)
    ns_helper.simulation_start()  # Force order of execution

    received = ns_helper.read_recvd_data(attime+1)
    ns_helper.simulation_start()
    print(received)
    print("**"*10)

numNodes = 10
ns_helper = ns_helper_new.NsHelper()

ns3Nodes = ns_helper.createNodes(numNodes)
ns3Interface = ns_helper.createInterface(ns3Nodes)

all_sockets = create_all_connections(numNodes, ns3Nodes, ns3Interface)
# print(all_sockets)

# fh, fm = ns_helper.monitoring_start(10.0)  # will move to class
order = [5, 2, 6, 4, 1, 3, 7, 2, 6, 5]

# sim_time = ns_helper.get_current_sim_time()+1
# oneround("hello", 1, 2, sim_time)
# time.sleep(2)
# sim_time = ns_helper.get_current_sim_time()+1
# oneround("from the other side", 1, 2, sim_time)

for ind in range(len(order) - 1):
    sim_time = ns_helper.get_current_sim_time() + 1
    oneround("hello", all_sockets, order[ind], order[ind+1], sim_time)

#ns_helper.monitoring_start()  # To get output from FlowMon

# ns_helper.monitoring_end(fh, fm)
ns_helper.simulation_end()

