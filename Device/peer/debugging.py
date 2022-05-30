import ns_helper
import numpy as np
import sys
import pickle
### Data

buffer = np.arange(3)
buffer = pickle.dumps(buffer)
numNodes = 10
### Creating Nodes
Initializer = ns_helper.ns_initializer(numNodes)

Nodes = Initializer.createNodes()

### Channel and Devices

Interfaces = Initializer.createInterface()

### Setting Source and Sink

source = Initializer.createSource(0)
sink = Initializer.createSink(2)

### Binding and connecting address

sinkAddress, anyAddress = Initializer.createSocketAddress()

Helper = ns_helper.nsHelper(sink, source, buffer)
Helper.act_as_server(sinkAddress)
Helper.act_as_client()


Helper.simulation_run()
Helper.simulation_end()
print(Helper.RecvData)