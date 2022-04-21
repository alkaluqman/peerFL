import ns.core
import ns.network
import ns.point_to_point
import ns.applications
import ns.wifi
import ns.mobility
import ns.csma
import ns.internet
import sys
import numpy as np

# // Network Topology
# //
# //   Wifi 10.1.3.0
# // C1   C2        AP
# //  *    *    *    *
# //  |    |    |    |    10.1.1.0
# // n5   n6   n7   n0 -------------- n1   n2   n3   n4
# //                   point-to-point  |    |    |    |
# //                                   *    *    *    *
# //                                   AP             S
# //                                     Wifi 10.1.2.0

# Command line arguments defined here
cmd = ns.core.CommandLine()
cmd.nWifi2 = 3
cmd.verbose = "True"
cmd.nWifi1 = 3
cmd.tracing = "False"

cmd.AddValue("nWifi2", "Number of wifi STA devices on left wifi network1")
cmd.AddValue("nWifi1", "Number of wifi STA devices on right wifi network 2")
cmd.AddValue("verbose", "Tell echo applications to log if true")
cmd.AddValue("tracing", "Enable pcap tracing")

cmd.Parse(sys.argv)

# Data type validation of input
nWifi2 = int(cmd.nWifi2)
verbose = cmd.verbose
nWifi1 = int(cmd.nWifi1)
tracing = cmd.tracing

ns.network.Packet.EnablePrinting(); # adding packet support
#packetSocket = ns.network.PacketSocketHelper()

# The underlying restriction of 18 is due to the grid position
# allocator's configuration; the grid layout will exceed the
# bounding box if more than 18 nodes are provided.
if nWifi1>18 or nWifi2>18:
	print ("nWifi should be 18 or less; otherwise grid layout exceeds the bounding box")
	sys.exit(1)

if verbose == "True":
	ns.core.LogComponentEnable("UdpEchoClientApplication", ns.core.LOG_LEVEL_INFO)
	ns.core.LogComponentEnable("UdpEchoServerApplication", ns.core.LOG_LEVEL_INFO)
	ns.core.LogComponentEnable("WifiHelper", ns.core.LOG_LEVEL_INFO)
	
# P2P network
p2pNodes = ns.network.NodeContainer() # NodeContainer for p2p
p2pNodes.Create(2) # 2 nodes, both will become AP

pointToPoint = ns.point_to_point.PointToPointHelper()
pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("5Mbps"))
pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

p2pDevices = pointToPoint.Install(p2pNodes) # Channel 


# ----- Wifi network 2 -----
# NodeContainer
wifi2StaNodes = ns.network.NodeContainer() 
wifi2StaNodes.Create(nWifi2)
wifi2ApNode = p2pNodes.Get(1)

# give packet socket powers to nodes.
# packetSocket.Install(wifi2StaNodes)
# packetSocket.Install(wifi2ApNode)

# Channel
channel2 = ns.wifi.YansWifiChannelHelper.Default() 
phy2 = ns.wifi.YansWifiPhyHelper()
phy2.SetChannel(channel2.Create())

# Install Wifi to form wifi devices
wifi = ns.wifi.WifiHelper()
wifi.SetRemoteStationManager("ns3::AarfWifiManager")
mac = ns.wifi.WifiMacHelper()
ssid = ns.wifi.Ssid ("ns-3-ssid")

mac.SetType ("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(ssid), "ActiveProbing", ns.core.BooleanValue(False))
wifi2StaDevices = wifi.Install(phy2, mac, wifi2StaNodes)

mac.SetType("ns3::ApWifiMac","Ssid", ns.wifi.SsidValue (ssid))
wifi2ApDevices = wifi.Install(phy2, mac, wifi2ApNode)

# Control mobility of wifi devices
mobility = ns.mobility.MobilityHelper()
mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(10.0), 
								"MinY", ns.core.DoubleValue (10.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0), 
                                 "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))
                                 
mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel", "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle (-50, 50, -50, 50)))
mobility.Install(wifi2StaNodes)

mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(wifi2ApNode)

# ----- Wifi network 1 -----
# NodeContainer
wifi1StaNodes = ns.network.NodeContainer() 
wifi1StaNodes.Create(nWifi1)
wifi1ApNode = p2pNodes.Get(0)

# give packet socket powers to nodes.
# packetSocket.Install(wifi1StaNodes)
# packetSocket.Install(wifi1ApNode)

# Channel
channel1 = ns.wifi.YansWifiChannelHelper.Default() 
phy1 = ns.wifi.YansWifiPhyHelper()
phy1.SetChannel(channel1.Create())

# Install Wifi to form wifi devices
wifi = ns.wifi.WifiHelper()
wifi.SetRemoteStationManager("ns3::AarfWifiManager")
mac = ns.wifi.WifiMacHelper()
ssid = ns.wifi.Ssid ("ns-3-ssid")

mac.SetType ("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(ssid), "ActiveProbing", ns.core.BooleanValue(False))
wifi1StaDevices = wifi.Install(phy1, mac, wifi1StaNodes)

mac.SetType("ns3::ApWifiMac","Ssid", ns.wifi.SsidValue (ssid))
wifi1ApDevices = wifi.Install(phy1, mac, wifi1ApNode)

# Control mobility of wifi devices
mobility = ns.mobility.MobilityHelper()
mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0), 
								"MinY", ns.core.DoubleValue (0.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0), 
                                 "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))
                                 
mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel", "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle (-50, 50, -50, 50)))
mobility.Install(wifi1StaNodes)

mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(wifi1ApNode)

# Set internet protocols
stack = ns.internet.InternetStackHelper()
stack.Install(wifi2ApNode)
stack.Install(wifi2StaNodes)
stack.Install(wifi1ApNode)
stack.Install(wifi1StaNodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
address.Assign(p2pDevices)

address.SetBase(ns.network.Ipv4Address("10.1.2.0"), ns.network.Ipv4Mask("255.255.255.0"))
wifi2Interfaces = address.Assign(wifi2StaDevices) # we only care about this cuz server will be installed here
address.Assign(wifi2ApDevices)

address.SetBase(ns.network.Ipv4Address("10.1.3.0"), ns.network.Ipv4Mask("255.255.255.0"))
address.Assign(wifi1StaDevices)
address.Assign(wifi1ApDevices)

# Assign server to right most node
echoServer = ns.applications.UdpEchoServerHelper(9)

serverApps = echoServer.Install(wifi2StaNodes.Get(nWifi2 - 1))

#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#reading packet data
#rcvd_msg = echoServer.HandleRead(serverApps.Get(0))
#  Create a packet sink to receive these packets
sink = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory",
#ns.network.InetSocketAddress(wifi2Interfaces.GetAddress(nWifi2 - 1), 9))
ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9))
print(sink)
##apps.Start(ns.core.Seconds(1))
sinkApp = sink.Install(wifi1StaNodes.Get(nWifi2 - 1)) # node5

##rcvd_msg = sink.
sinkApp.Start(ns.core.Seconds(1.0))
sinkApp.Stop(ns.core.Seconds(10.0))

#serverApps.Start(ns.core.Seconds(1.0))
#serverApps.Stop(ns.core.Seconds(10.0))

#----------------------------------------------------------------------#
#p = ns.applications.PacketSink()
#----------------------------------------------------------------------#

# Assign client1 to node 5
echoClient1 = ns.applications.UdpEchoClientHelper(wifi2Interfaces.GetAddress(nWifi2 - 1), 9)
echoClient1.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
echoClient1.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds (1.0)))
echoClient1.SetAttribute("PacketSize", ns.core.UintegerValue(1024))

clientApps1 = echoClient1.Install(wifi1StaNodes.Get(nWifi1 - 1)) # node5
clientApps1.Start(ns.core.Seconds(7.0))
clientApps1.Stop(ns.core.Seconds(10.0))

#creating packet data
msg1 = "Hello World"
echoClient1.SetFill(clientApps1.Get(0), msg1)

# Assign client2 to node 7
echoClient2 = ns.applications.UdpEchoClientHelper(wifi2Interfaces.GetAddress(nWifi2 - 1), 9)
echoClient2.SetAttribute("PacketSize", ns.core.UintegerValue(512))
echoClient2.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
echoClient2.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds (1.0)))

clientApps2 = echoClient2.Install(wifi1StaNodes.Get(1)) # node7
clientApps2.Start(ns.core.Seconds(2.0))
clientApps2.Stop(ns.core.Seconds(5.0))

#creating packet data
rand_int2 = np.random.randint(10,90,(4,5)) # random numpy array of shape (4,5)
echoClient2.SetFill(clientApps2.Get(0), rand_int2)

# Abracadabra
ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

ns.core.Simulator.Stop(ns.core.Seconds(10.0))

if tracing == "True":
	phy1.SetPcapDataLinkType(phy.DLT_IEEE802_11_RADIO)
	pointToPoint.EnablePcapAll ("third")
	phy1.EnablePcap ("third", apDevices.Get (0))
	#csma.EnablePcap ("third", csmaDevices.Get (0), True)

#ascii = ns.network.AsciiTraceHelper();
#stream = ascii.CreateFileStream("wtf.tr");
#phy1.EnableAsciiAll(stream);
#phy2.EnableAsciiAll(stream);





###########################################################################

import numpy as np
import random
import cv2
import os
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from keras import Sequential
from keras import applications

from tensorflow.keras.applications.vgg16 import VGG16
'Import the datagenerator to augment images'
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau

'Lastly import the final layers that will be added on top of the base model'
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout

'Import to_categorical from the keras utils package to one hot encode the labels'

from keras.datasets import cifar10

def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}



def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


class SimpleMLP:
    @staticmethod
    def build(shape, classes,only_digits=True):
        base_model_1 = VGG16(include_top=False,input_shape=(32,32,3),classes=y_train.shape[1])
        model_1= Sequential()
        model_1.add(base_model_1) #Adds the base model (in this case vgg19 to model_1)
        model_1.add(Flatten()) #Since the output before the flatten layer is a matrix we have to use this function to get a vector of the form nX1 to feed it into the fully connected layers
        #Add the Dense layers along with activation and batch normalization
        model_1.add(Dense(1024,activation=('relu'),input_dim=512))
        model_1.add(Dense(512,activation=('relu'))) 
        model_1.add(Dense(256,activation=('relu'))) 
        #model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
        model_1.add(Dense(128,activation=('relu')))
        #model_1.add(Dropout(.2))
        model_1.add(Dense(10,activation=('softmax'))) #This is the classification layer
        return model_1

	
 
 
    

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(1 * weight[i])
    return weight
def scale_model_weights2(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(1 * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad
def test_model_mid(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    return acc, loss

def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('acc: {:.3%} | loss: {}'.format(acc, loss))
    return acc, loss
    
 
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm
	
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


X_train, y_train, X_test, y_test = load_dataset()
X_train , X_test = prep_pixels(X_train, X_test)
print(X_train.shape)

#create clients
clients = create_clients(X_train, y_train, num_clients=10, initial='client')

#process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
    
#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(32)
smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(3072, 10) 
lr = 0.01 
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, 
                decay=lr , 
                momentum=0.9
               ) 
SGD_model.compile(loss=loss, 
              optimizer=optimizer, 
              metrics=metrics)

# fit the SGD training data to model
_ = SGD_model.fit(SGD_dataset, epochs=20, verbose=1)

#test the SGD global model and print out metrics
for(X_test, Y_test) in test_batched:
        SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)
###########################################################################




print("Running Simulation");
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
print("Done");
