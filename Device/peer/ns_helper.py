import ns.core
import ns.internet
import ns.applications
import ns.mobility
import ns.network
import ns.wifi
import pickle

class ns_initializer():
    def __init__(self, numNodes, mobility = True, address = "10.1.1.0"):
        self.numNodes = numNodes
        self.Nodes = None
        self.Interfaces = None
        self.Devices = None
        self.address = address
        self.mobility = mobility

    def createNodes(self):
        Nodes = ns.network.NodeContainer()
        Nodes.Create(self.numNodes)
        self.Nodes = Nodes
        return Nodes

    def createInterface(self):
        wifi = ns.wifi.WifiHelper()

        Phy = ns.wifi.YansWifiPhyHelper.Default()
        Channel = ns.wifi.YansWifiChannelHelper.Default()
        Phy.SetChannel(Channel.Create())

        ssid = ns.wifi.Ssid("ns-3-ssid")
        wifi.SetRemoteStationManager("ns3::ArfWifiManager")
        Mac = ns.wifi.WifiMacHelper()

        Mac.SetType("ns3::AdhocWifiMac", "Ssid", ns.wifi.SsidValue(ssid))
        Devices = wifi.Install(Phy, Mac, self.Nodes)
        self.Devices = Devices
        mobility = ns.mobility.MobilityHelper()
        if self.mobility:
            mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0), 
                                            "MinY", ns.core.DoubleValue (0.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0), 
                                            "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))

        else:
            mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")

        mobility.Install(self.Nodes)

        stack = ns.internet.InternetStackHelper()
        stack.Install(self.Nodes)

        address = ns.internet.Ipv4AddressHelper()
        address.SetBase(
            ns.network.Ipv4Address(self.address),
            ns.network.Ipv4Mask("255.255.255.0")
        )
        Interface = address.Assign(Devices)
        self.Interfaces = Interface
        return Interface

    def createSource(self, sender):
        source = ns.network.Socket.CreateSocket(
            self.Nodes.Get(sender),
            ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
        )
        return source
    
    def createSink(self, receiver):
        sink = ns.network.Socket.CreateSocket(
            self.Nodes.Get(receiver),
            ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
        )
        self.receiver = receiver
        return sink

    def createSocketAddress(self):
        sinkAddress = ns.network.InetSocketAddress(self.Interfaces.GetAddress(self.receiver), 9)
        anyAddress = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)

        return sinkAddress, anyAddress
    
class nsHelper():
    def __init__(self, sink, source, buffer, size = 1024):
        self.sink = sink
        self.source = source
        self.RecvData = None
        self.buffer = buffer
        self.size = size

    def act_as_client(self, address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)):
        self.sink.Bind(address)

    def act_as_server(self, sinkAddress):
        self.source.Connect(sinkAddress)
    
    def sendPacket(self, socket):
        print("Sending", ns.core.Simulator.Now())
        socket.Send(ns.network.Packet(str(self.buffer)[2:-1], self.size))
    
    def receivePacket(self, socket):
        print("Recieving", ns.core.Simulator.Now())
        tmp = socket.Recv().GetString().encode().decode("unicode_escape").encode("raw_unicode_escape")
        self.RecvData = pickle.loads(tmp)

    def simulation_run(self, time = 0.0):
        self.sink.SetRecvCallback(self.receivePacket)
        ns.core.Simulator.Schedule(
        ns.core.Seconds(time), self.sendPacket, self.source, 
        )
        ns.core.Simulator.Run()
    
    def simulation_end(self):
        ns.core.Simulator.Destroy()


