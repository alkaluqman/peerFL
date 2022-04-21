import ns.core
import ns.network
import ns.point_to_point
import ns.applications
import ns.wifi
import ns.mobility
import ns.csma
import ns.internet
import sys

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

# The underlying restriction of 18 is due to the grid position
# allocator's configuration; the grid layout will exceed the
# bounding box if more than 18 nodes are provided.
if nWifi1>18 or nWifi2>18:
	print ("nWifi should be 18 or less; otherwise grid layout exceeds the bounding box")
	sys.exit(1)

if verbose == "True":
	ns.core.LogComponentEnable("UdpEchoClientApplication", ns.core.LOG_LEVEL_INFO)
	ns.core.LogComponentEnable("UdpEchoServerApplication", ns.core.LOG_LEVEL_INFO)
	
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
serverApps.Start(ns.core.Seconds(1.0))
serverApps.Stop(ns.core.Seconds(10.0))

# Assign client1 to node 5
echoClient1 = ns.applications.UdpEchoClientHelper(wifi2Interfaces.GetAddress(nWifi2 - 1), 9)
echoClient1.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
echoClient1.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds (1.0)))
echoClient1.SetAttribute("PacketSize", ns.core.UintegerValue(1024))

clientApps1 = echoClient1.Install(wifi1StaNodes.Get(nWifi1 - 1)) # node5
clientApps1.Start(ns.core.Seconds(7.0))
clientApps1.Stop(ns.core.Seconds(10.0))

# Assign client2 to node 7
echoClient2 = ns.applications.UdpEchoClientHelper(wifi2Interfaces.GetAddress(nWifi2 - 1), 9)
echoClient2.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
echoClient2.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds (1.0)))
echoClient2.SetAttribute("PacketSize", ns.core.UintegerValue(512))

clientApps2 = echoClient2.Install(wifi1StaNodes.Get(1)) # node7
clientApps2.Start(ns.core.Seconds(2.0))
clientApps2.Stop(ns.core.Seconds(5.0))


# Abracadabra
ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

ns.core.Simulator.Stop(ns.core.Seconds(10.0))

if tracing == "True":
	phy.SetPcapDataLinkType(phy.DLT_IEEE802_11_RADIO)
	pointToPoint.EnablePcapAll ("third")
	phy.EnablePcap ("third", apDevices.Get (0))
	csma.EnablePcap ("third", csmaDevices.Get (0), True)

ns.core.Simulator.Run()
ns.core.Simulator.Destroy()

