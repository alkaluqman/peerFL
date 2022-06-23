import ns.core
import ns.internet
import ns.applications
import ns.mobility
import ns.network
import ns.wifi
import pickle
import sys
import ns.flow_monitor

class ns_initializer():
    def __init__(self, numNodes, mobility=True, address="10.1.1.0"):
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

    def createInterface(self, loss=False, PowerStart=1, PowerEnd=1, Freq=5.180e9):
        wifi = ns.wifi.WifiHelper()

        Phy = ns.wifi.YansWifiPhyHelper.Default()
        Channel = ns.wifi.YansWifiChannelHelper.Default()
        if loss:
            Channel.AddPropagationLoss("ns3::FriisPropagationLossModel",
                                       "Frequency", ns.core.DoubleValue(Freq))
            Channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel")
            Phy.SetChannel(Channel.Create())
            Phy.Set("TxPowerStart", ns.core.DoubleValue(PowerStart))
            Phy.Set("TxPowerEnd", ns.core.DoubleValue(PowerEnd))
        else:
            Phy.SetChannel(Channel.Create())

        ssid = ns.wifi.Ssid("ns-3-ssid")
        wifi.SetRemoteStationManager("ns3::ArfWifiManager")
        Mac = ns.wifi.WifiMacHelper()

        Mac.SetType("ns3::AdhocWifiMac", "Ssid", ns.wifi.SsidValue(ssid))
        Devices = wifi.Install(Phy, Mac, self.Nodes)
        self.Devices = Devices
        mobility = ns.mobility.MobilityHelper()
        if self.mobility:
            mobility.SetPositionAllocator("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0),
                                          "MinY", ns.core.DoubleValue(0.0), "DeltaX", ns.core.DoubleValue(5.0),
                                          "DeltaY", ns.core.DoubleValue(10.0),
                                          "GridWidth", ns.core.UintegerValue(3), "LayoutType",
                                          ns.core.StringValue("RowFirst"))

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
    def __init__(self, sink = None, source = None, size=1024, DataRate="512kb/s", verbose=False):
        self.sink = sink
        self.source = source
        self.RecvData = []
        self.netData = []
        self.Start = True
        self.toRecv = -1
        self.DataRate = ns.network.DataRate(DataRate)
        self.size = size
        self.totData = []
        self.RecvPacket = 0
        self.verbose = verbose
        self.flowmon_helper = None
        self.flowmonitor = None

    def act_as_client(self, address=ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)):
        self.sink.Bind(address)

    def act_as_server(self, sinkAddress):
        self.source.Connect(sinkAddress)

    def makePackets(self, data):
        data = str(pickle.dumps(data))[2:-1]

        size = sys.getsizeof(data)
        self.numPackets = int(size / self.size)
        p = ns.network.Packet(data, size)
        self.packets = [ns.network.Packet(str(pickle.dumps(self.numPackets + 1))[2:-1], self.size)]
        for i, start in enumerate(range(0, size, self.size)):
            if (i < self.numPackets):
                self.packets.append(p.CreateFragment(start, self.size))
        self.packets.append(p.CreateFragment(start, size - start))
        # self.split_data = [data[i:i+self.size] for i in range(0, len(data), self.size)]
        self.numPackets = len(self.packets)
        self.totData.append(self.packets)

    def get_packets(self):
        if self.totData:
            self.packets = self.totData.pop(0)  
            self.numPackets = len(self.packets)  

    def sendPacket(self, socket):
        if self.packets:
            if self.verbose:
                print("Sending", ns.core.Simulator.Now().GetSeconds())
            socket.Send(self.packets.pop(0))
            
        else:
            socket.Close()

    def receivePacket(self, socket):
        if self.verbose:
            print("Recieving", ns.core.Simulator.Now().GetSeconds())
        tmp = socket.Recv(maxSize = self.size, flags = 0)
        self.RecvData.append(tmp)
        self.RecvPacket += 1
        #print(self.toRecv)
        if(self.RecvPacket == self.toRecv or self.Start):
            if(self.RecvPacket == self.toRecv):
                self.getRecvData()
                print("yes received total")
                self.Start = True
                self.toRecv = -1
            elif self.Start:
                self.getRecvData()
                self.toRecv = self.netData.pop(-1)
                self.Start = False
            self.RecvPacket = 0
        

    def getRecvData(self):
        
        #self.RecvData = [self.RecvData[i][:len(item)] for i, item in enumerate(self.split_data)]
        for packets in self.RecvData[1:]:
            self.RecvData[0].AddAtEnd(packets)
        self.RecvData = self.RecvData[0].GetString()
        self.netData.append(pickle.loads("".join(self.RecvData).encode().decode("unicode_escape").encode("raw_unicode_escape")))
        self.RecvData = []
        #return pickle.loads("".join(self.RecvData).encode().decode("unicode_escape").encode("raw_unicode_escape"))

    def init_send(self, time):
        ns.core.Simulator.Schedule(
            ns.core.Seconds(time), self.get_packets
        )
        for _ in range(self.numPackets):
            # delta = ns.core.Simulator.Now() - time
            ns.core.Simulator.Schedule(
                ns.core.Seconds(time), self.sendPacket, self.source
            )
            time += self.size * 8.0 / float(self.DataRate.GetBitRate())
        ##print(time)
        #ns.core.Simulator.Schedule(
        #    ns.core.Seconds(time + 0.01), self.getRecvData
        #)
    def print_stats(self, os, st):
        print("  Tx Bytes: ", st.txBytes, file=os)
        print("  Rx Bytes: ", st.rxBytes, file=os)
        print("  Tx Packets: ", st.txPackets, file=os)
        print("  Rx Packets: ", st.rxPackets, file=os)
        print("  Lost Packets: ", st.lostPackets, file=os)
        if st.rxPackets > 0:
            print("  Mean{Delay}: ", (st.delaySum.GetSeconds() / st.rxPackets), file=os)
            print("  Mean{Jitter}: ", (st.jitterSum.GetSeconds() / (st.rxPackets - 1)), file=os)
            print("  Mean{Hop Count}: ", float(st.timesForwarded) / st.rxPackets + 1, file=os)

        for reason, drops in enumerate(st.packetsDropped):
            print("  Packets dropped by reason %i: %i" % (reason, drops), file=os)
        # for reason, drops in enumerate(st.bytesDropped):
        #    print "Bytes dropped by reason %i: %i" % (reason, drops)

    def simulation_run(self, time=0.0):
        self.init_send(time)
        self.sink.SetRecvCallback(self.receivePacket)

    def simulation_start(self):
        #flow monitoring statistics
        #self.flowmon_helper = ns.flow_monitor.FlowMonitorHelper()
        #self.flowmonitor = self.flowmon_helper.InstallAll()
        #ns.core.Simulator.Stop(ns.core.Seconds(30.0))  # FlowMonitor will not work without this
        ns.core.Simulator.Run()

    def simulation_end(self):
        #classifier = self.flowmon_helper.GetClassifier()
        #for flow_id, flow_stats in self.flowmonitor.GetFlowStats():
        #    t = classifier.FindFlow(flow_id)
        #    proto = {6: 'TCP', 17: 'UDP'}[t.protocol]
        #    print("FlowID: %i (%s %s/%s --> %s/%i)" % (
        #    flow_id, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
        #    self.print_stats(sys.stdout, flow_stats)

        ns.core.Simulator.Destroy()
        # self.monitor.SerializeToXmlFile("nodestats.xml", True, True)
        

