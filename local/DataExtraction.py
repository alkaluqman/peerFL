import ns.network
import ns.core
import ns.applications
import ns.point_to_point
import ns.internet
import numpy as np

buffer = str(np.array([1,2,3]).tobytes())[2:-1]

def sendPacket(socket):
    print("Sending", ns.core.Simulator.Now())
    socket.Send(ns.network.Packet(buffer, 1024))

def receivePacket(socket):
    print("received", ns.core.Simulator.Now())
    print("printing contents")
    tmp = socket.Recv().GetString().encode().decode("unicode_escape").encode("raw_unicode_escape")
    print(np.frombuffer(tmp, np.int64))
nodes = ns.network.NodeContainer()
nodes.Create(2)

p2p = ns.point_to_point.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("5Mbps"))
p2p.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

devices = p2p.Install(nodes)

stack = ns.internet.InternetStackHelper()
stack.Install(nodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(
    ns.network.Ipv4Address("10.1.1.0"),
    ns.network.Ipv4Mask("255.255.255.0")
)

interface = address.Assign(devices)

source = ns.network.Socket.CreateSocket(
    nodes.Get(0),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sink = ns.network.Socket.CreateSocket(
    nodes.Get(1),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sinkAddress = ns.network.InetSocketAddress(interface.GetAddress(1), 9)
anyAddress = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)

sink.Bind(anyAddress)
source.Connect(sinkAddress)

sink.SetRecvCallback(receivePacket)

ns.core.Simulator.Schedule(
    ns.core.Seconds(0.0), sendPacket, source, 
)

ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
