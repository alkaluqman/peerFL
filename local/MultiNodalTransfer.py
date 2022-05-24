import ns.core
import ns.internet
import ns.applications
import ns.point_to_point
import ns.network
import ns.csma
import numpy as np
import sys

buffer = str(np.arange(3).tobytes())[2:-1]

def sendPacket(socket):
    print("Sending", ns.core.Simulator.Now())
    socket.Send(ns.network.Packet(buffer, 1024))

def receivePacket(socket, dtype = np.int64):
    print("Recieving", ns.core.Simulator.Now())
    print("Shoiwng received contents")
    tmp = socket.Recv().GetString().encode().decode("unicode_escape").encode("raw_unicode_escape")
    print(np.frombuffer(tmp, dtype))

###
cmd =ns.core.CommandLine()
cmd.nodes = 9
cmd.receiver = 2
cmd.sender = 0
cmd.AddValue("nodes", "Number of csma Nodes")
cmd.AddValue("receiver", "Receiver Node")
cmd.AddValue("sender", "Sender Node")
cmd.Parse(sys.argv)

numNodes = int(cmd.nodes)
recv = int(cmd.receiver)
sender = int(cmd.sender)
csmaNodes = ns.network.NodeContainer()
csmaNodes.Create(numNodes)

csma = ns.csma.CsmaHelper()
csma.SetChannelAttribute("DataRate", ns.core.StringValue("10Mbps"))
csma.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

csmaDevices = csma.Install(csmaNodes)

stack = ns.internet.InternetStackHelper()
stack.Install(csmaNodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(
    ns.network.Ipv4Address("10.1.1.0"),
    ns.network.Ipv4Mask("255.255.255.0")
)
csmaInterfaces = address.Assign(csmaDevices)

source = ns.network.Socket.CreateSocket(
    csmaNodes.Get(sender),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sink = ns.network.Socket.CreateSocket(
    csmaNodes.Get(recv),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sinkAddress = ns.network.InetSocketAddress(csmaInterfaces.GetAddress(recv), 9)
anyAddress = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)

sink.Bind(anyAddress)
source.Connect(sinkAddress)
sink.SetRecvCallback(receivePacket)
csma.EnablePcap ("test/second", csmaDevices)
ns.core.Simulator.Schedule(
    ns.core.Seconds(0.0), sendPacket, source, 
)

ns.core.Simulator.Run()
ns.core.Simulator.Destroy()