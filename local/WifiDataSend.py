
import ns.core
import ns.internet
import ns.applications
import ns.mobility
import ns.network
import ns.wifi
import numpy as np
import sys

### Data

buffer = str(np.arange(3).tobytes())[2:-1]

### Helper Functions for Sending and receiving 

def sendPacket(socket):
    print("Sending", ns.core.Simulator.Now())
    socket.Send(ns.network.Packet(buffer, 1024))

def receivePacket(socket, dtype = np.int64):
    print("Recieving", ns.core.Simulator.Now())
    print("Shoiwng received contents")
    tmp = socket.Recv().GetString().encode().decode("unicode_escape").encode("raw_unicode_escape")
    print(np.frombuffer(tmp, dtype))

### Creating Nodes
cmd =ns.core.CommandLine()
cmd.nodes = 10
cmd.recv = 2
cmd.sndr = 0
cmd.AddValue("nodes", "Number of csma Nodes")
cmd.AddValue("recv", "Index of client(0-Indexed)")
cmd.AddValue("sndr", "Index of client(0-Indexed)")
cmd.Parse(sys.argv)

numNodes = int(cmd.nodes)
receiver = int(cmd.recv)
sender = int(cmd.sndr)

assert sender != receiver, "Sender and Receiver cant be same."

Nodes = ns.network.NodeContainer()
Nodes.Create(numNodes)

### Channel and Devices

wifi = ns.wifi.WifiHelper()

Phy = ns.wifi.YansWifiPhyHelper.Default()
Channel = ns.wifi.YansWifiChannelHelper.Default()
Phy.SetChannel(Channel.Create())

ssid = ns.wifi.Ssid("ns-3-ssid")
wifi.SetRemoteStationManager("ns3::ArfWifiManager")
Mac = ns.wifi.WifiMacHelper()

# setup stas.


Mac.SetType("ns3::AdhocWifiMac")
Devices = wifi.Install(Phy, Mac, Nodes)

### Mobility

mobility = ns.mobility.MobilityHelper()
mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0), 
								"MinY", ns.core.DoubleValue (0.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0), 
                                 "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))
mobility.Install(Nodes)

### Stack and Address

stack = ns.internet.InternetStackHelper()
stack.Install(Nodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(
    ns.network.Ipv4Address("10.1.1.0"),
    ns.network.Ipv4Mask("255.255.255.0")
)
StaInterface = address.Assign(Devices)
### Setting Source and Sink

source = ns.network.Socket.CreateSocket(
    Nodes.Get(sender),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sink = ns.network.Socket.CreateSocket(
    Nodes.Get(receiver),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

### Binding and connecting address

sinkAddress = ns.network.InetSocketAddress(StaInterface.GetAddress(receiver), 9)
anyAddress = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)

sink.Bind(anyAddress)
source.Connect(sinkAddress)

sink.SetRecvCallback(receivePacket)
ns.core.Simulator.Schedule(
    ns.core.Seconds(0.0), sendPacket, source, 
)

ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
