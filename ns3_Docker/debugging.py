#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ns_helper
import numpy as np
import sys
import pickle
import time
import ns.core
#import tensorflow as tf

### Data
data = np.arange(3)

numNodes = 10
### Creating Nodes
Initializer = ns_helper.ns_initializer(numNodes)

Nodes = Initializer.createNodes()
print(Nodes)

### Channel and Devices

Interfaces = Initializer.createInterface()
print(Interfaces)

### Send from A to B

#def train():
#    pass



Helper = ns_helper.nsHelper(size= 512, verbose=True)
def send(from_node, to_node, attime=0.0):
    source = Initializer.createSource(from_node)  # from_node is just the number, eg: 1
    sink = Initializer.createSink(to_node)
    sinkAddress, _ = Initializer.createSocketAddress()

    Helper.sink = sink
    Helper.source = source
    #Model Training at source
    #time.sleep(t) We can get the value of T

    Helper.makePackets(data) # Sends model data to sink
    Helper.act_as_server(sinkAddress)
    Helper.act_as_client()

    Helper.simulation_run(attime)
    # print(Helper.getRecvData())

time = 0
### Define runs
order = [5, 2, 6, 4, 1, 3, 7, 2, 6, 5]
for ind in range(len(order) - 1):
    #train(Node[ind]) We get time for training
    t = 1 ## Hardcoded for now
    send(order[ind], order[ind+1], time + t)
    Helper.simulation_start()
    time+= ns.core.Simulator.Now().GetSeconds()

    
    



### Start all simulation runs
#Helper.simulation_start()
#Helper.simulation_end()
Helper.simulation_end()
print(Helper.netData)