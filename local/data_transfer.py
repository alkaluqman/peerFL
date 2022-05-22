import ns.core
import ns.point_to_point
import ns.internet
import ns.applications
import ns.network
import ns.wifi
import ns.mobility
import sys


'''
We are building the following topology
      10.1.1.0          10.1.2.0
n0 ----------------- n1  n2  n3  n4  
    point-to-point    |   |   |   |
                      Ap  Sta Sta Sta


'''

def make_apps(i):
    Client = ns.applications.UdpEchoClientHelper(ApInterface.GetAddress(0), 9)
    Client.SetAttribute("MaxPackets", ns.core.UintegerValue(1000000))
    Client.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(1.0)))
    Client.SetAttribute("PacketSize", ns.core.UintegerValue(1024000))
    clientApps = Client.Install(StaNodes.Get(i))
    clientApps.Start(ns.core.Seconds(1.0))
    clientApps.Stop(ns.core.Seconds(10.0))
    return

### Command Line inputs

cmd = ns.core.CommandLine()
cmd.tracing = "True"
cmd.numNodes = 4
cmd.udp = "True"
cmd.simulationTime = 10 #seconds
cmd.distance = 1.0 #meters
cmd.frequency = 5.0 #whether 2.4 or 5.0 GHz

cmd.AddValue("frequency", "Whether working in the 2.4 or 5.0 GHz band (other values gets rejected)")
cmd.AddValue("distance", "Distance in meters between the station and the access point")
cmd.AddValue("simulationTime", "Simulation time in seconds")
cmd.AddValue("udp", "UDP if set to True, TCP otherwise")
cmd.AddValue("numNodes", "Number of Nodes")

cmd.Parse(sys.argv)

tracing = cmd.tracing
numNodes = int(cmd.numNodes);udp = cmd.udp
simulationTime = float(cmd.simulationTime)
distance = float(cmd.distance)
frequency = float(cmd.frequency)


ns.core.LogComponentEnable("UdpEchoClientApplication", ns.core.LOG_LEVEL_INFO)
ns.core.LogComponentEnable("UdpEchoServerApplication", ns.core.LOG_LEVEL_INFO)

### Creating Nodes
p2pNodes = ns.network.NodeContainer()
p2pNodes.Create(2)

StaNodes = ns.network.NodeContainer()
StaNodes.Create(numNodes - 1)
ApNode = p2pNodes.Get(1)

### Setting P2P and Wifi

p2p = ns.point_to_point.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("5Mbps"))
p2p.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

p2pDevices = p2p.Install(ns.network.NodeContainer(p2pNodes))
channel = ns.wifi.YansWifiChannelHelper.Default()
phy = ns.wifi.YansWifiPhyHelper()
phy.SetChannel(channel.Create())

wifi = ns.wifi.WifiHelper()
wifi.SetRemoteStationManager("ns3::AarfWifiManager")
mac = ns.wifi.WifiMacHelper()
ssid = ns.wifi.Ssid ("ns-3-ssid")

mac.SetType("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(ssid), "ActiveProbing", ns.core.BooleanValue(False))
StaDevices = wifi.Install(phy, mac, StaNodes)

mac.SetType("ns3::ApWifiMac","Ssid", ns.wifi.SsidValue (ssid))
ApDevice = wifi.Install(phy, mac, ApNode)



### Giving Mobility to devices

mobility = ns.mobility.MobilityHelper()

mobility.SetPositionAllocator("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0), 
								"MinY", ns.core.DoubleValue (0.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0), 
                                "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))
                                 
mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel", "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle (-50, 50, -50, 50)))
mobility.Install(StaNodes)


mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(ApNode)

### Internet Stack

stack = ns.internet.InternetStackHelper()
stack.Install(p2pNodes.Get(0))
stack.Install(StaNodes)
stack.Install(ApNode)

### Setting IP Address

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
p2pInterface = address.Assign(p2pDevices)

address.SetBase(ns.network.Ipv4Address("10.1.2.0"), ns.network.Ipv4Mask("255.255.255.0"))
StaInterface = address.Assign(StaDevices)
ApInterface = address.Assign(ApDevice)

### Setting Applications

if udp == "True":

    Server = ns.applications.UdpEchoServerHelper(9)
    serverApps = Server.Install(ApNode)
    serverApps.Start(ns.core.Seconds(1.0))
    serverApps.Stop(ns.core.Seconds(10.0))

    Client1 = ns.applications.UdpEchoClientHelper(ApInterface.GetAddress(0), 9)
    Client1.SetAttribute("MaxPackets", ns.core.UintegerValue(1000000))
    Client1.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(1.0)))
    Client1.SetAttribute("PacketSize", ns.core.UintegerValue(1024000))
    clientApps1 = Client1.Install(StaNodes)

    clientApps1.Start(ns.core.Seconds(1.0))
    clientApps1.Stop(ns.core.Seconds(10.0))

''''
Clients = []
Apps = []
for i in range(numNodes-1):
    exec(f"Client_{i}, clientApps_{i} = 0, 0")
    exec(f"Clients.append(Client_{i})")
    exec(f"Apps.append(clientApps_{i})")

for i in range(numNodes-1):
    Clients[i] = ns.applications.UdpEchoClientHelper(ApInterface.GetAddress(0), 9)
    Clients[i].SetAttribute("MaxPackets", ns.core.UintegerValue(1000000))
    Clients[i].SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(1.0)))
    Clients[i].SetAttribute("PacketSize", ns.core.UintegerValue(1024000))
    Apps[i] = Clients[i].Install(StaNodes.Get(i))

    Apps[i].Start(ns.core.Seconds(1.0))
    Apps[i].Stop(ns.core.Seconds(10.0))

'''

Client1.SetFill(clientApps1.Get(0), "Yes")
Client1.SetFill(clientApps1.Get(1), "Yes1")
Client1.SetFill(clientApps1.Get(2), "Yes2")



ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

ns.core.Simulator.Stop(ns.core.Seconds(simulationTime + 1))



if tracing == "True":
    p2p.EnablePcap("tests/p2ptest", p2pDevices)
    phy.EnablePcap("tests/APtest", ApDevice)
    phy.EnablePcap("tests/Statest", StaDevices)

ns.core.Simulator.Run()
ns.core.Simulator.Destroy()


