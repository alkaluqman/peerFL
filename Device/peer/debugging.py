#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import ns_helper
import numpy as np
import sys
import pickle
#import tensorflow as tf

### Data
data = np.arange(100)

numNodes = 10
### Creating Nodes
Initializer = ns_helper.ns_initializer(numNodes)

Nodes = Initializer.createNodes()

### Channel and Devices

Interfaces = Initializer.createInterface()

### Setting Source and Sink

source = Initializer.createSource(1)
sink = Initializer.createSink(2)

### Binding and connecting address

sinkAddress, anyAddress = Initializer.createSocketAddress()

Helper = ns_helper.nsHelper(sink, source,512, verbose=True)
Helper.makePackets(data)
Helper.act_as_server(sinkAddress)
Helper.act_as_client()


Helper.simulation_run()
Helper.simulation_end()
print(Helper.getRecvData())
#print(len([len(item) for item in Helper.RecvData]))
#print(len([len(item) for item in Helper.split_data]))
#Helper.RecvData = [Helper.RecvData[i][:len(item)] for i, item in enumerate(Helper.split_data)]
#print([True if Helper.RecvData[i] == Helper.split_data[i] else False for i in range(len(Helper.split_data))])
#print(pickle.loads("".join(Helper.split_data).encode().decode("unicode_escape").encode("raw_unicode_escape")))


