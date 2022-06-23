#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ns_helper_new
import numpy as np
import sys
import pickle
import time
#import tensorflow as tf


def oneround(data, from_node, to_node, attime=0.0):
    ns_helper.send_data(from_node, data, attime)

    received = ns_helper.recv_data(to_node)
    ns_helper.simulation_start()  # Force order of execution

    received = ns_helper.read_recvd_data(attime+0.5)
    ns_helper.simulation_start()
    print(received)


numNodes = 10
ns_helper = ns_helper_new.NsHelper()

ns3Nodes = ns_helper.createNodes(numNodes)
ns3Interface = ns_helper.createInterface(ns3Nodes)

in_socket = ns_helper.act_as_client(ns3Nodes, 2)
out_socket = ns_helper.act_as_server(ns3Nodes, ns3Interface, 1, 2)

# fh, fm = ns_helper.monitoring_start(10.0)  # will move to class
oneround("hello", out_socket, in_socket, 0.0)
time.sleep(2)
oneround("from the other side", out_socket, in_socket, 5.0)

#ns_helper.monitoring_start()  # To get output from FlowMon

# ns_helper.monitoring_end(fh, fm)
ns_helper.simulation_end()


