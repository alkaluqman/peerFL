import ns.core
import ns.network
import ns.wifi
import ns.internet
import ns.applications
import ns.mobility
#print(*dir(ns.mobility.ListPositionAllocator), sep = '\n')


numNodes = 2
distance = 50.0
time = 10.0
enablePcap = True

ApNode = ns.network.NodeContainer()
ApNode.Create(1)

StaNodes = ns.network.NodeContainer()
StaNodes.Create(numNodes - 1)

phy = ns.wifi.YansWifiPhyHelper()
Channel = ns.wifi.YansWifiChannelHelper()
Channel.AddPropagationLoss("ns3::FriisPropagationLossModel",
                                      "Frequency", ns.core.DoubleValue(5.180e9))
Channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel")
phy.SetChannel(Channel.Create())
phy.Set("TxPowerStart", ns.core.DoubleValue(1))
phy.Set("TxPowerEnd", ns.core.DoubleValue(1))

wifi = ns.wifi.WifiHelper()
mac = ns,wifi.WifiMacHelper()
ssid = ns.wifi.Ssid ("ns-3-ssid")

wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode", ns.core.StringValue("OfdmRate54Mbps"))

mac.SetType("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(ssid))
staDevices = wifi.Install(phy, mac, StaNodes)
mac.SetType("ns3::ApWifiMac", "Ssid", ns.wifi.SsidValue(ssid))
apDevices = wifi.Install(phy, mac, ApNode)

mobility = ns.mobility.MobilityHelper()
positionAlloc = ns.mobility.ListPositionAllocator()
positionAlloc.Add(ns.core.Vector(0.0, 0.0, 0.0))
positionAlloc.Add(ns.core.vector(distance, 0.0, 0.0))
mobility.SetPositionAllocator(positionAlloc)

mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(StaNodes)
mobility.Install(ApNode)

stack = ns.internet.InternetStackHelper();
stack.Install(ApNode)
stack.Install(StaNodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
StaInterfaces = address.Assign(staDevices)
ApInterface = address.Assign(apDevices)

server = ns.applications.UdpServerHelper()
serverApp = server.Install(StaNodes.Get(0))
serverApp.Start(ns.core.Seconds(0.0))
serverApp.Stop(ns.core.Seconds(time))

client = ns.applications.UdpClientHelper()
client.SetAttribute("MaxPackets", ns.core.UintegerValue(100000))
client.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(1.0)))
client.SetAttribute("PacketSize", ns.core.UintegerValue(1024))
clientApp = client.Install(ApNode.Get(0))
clientApp.Start(ns.core.Seconds(0.0))
clientApp.Stop(ns.core.Seconds(time))

if(enablePcap):
    phy.EnablePcap("tests/APtest", apDevices)
    phy.EnablePcap("tests/Statest", staDevices)

ns.core.Simulator.run()
ns.core.Simulator.Destroy()