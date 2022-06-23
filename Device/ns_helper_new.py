import ns.core
import ns.internet
import ns.applications
import ns.mobility
import ns.network
import ns.wifi
import pickle
import sys
import ns.flow_monitor
import time


class NsHelper():
    def __init__(self):
        self.rcvd_data = None

    def createNodes(self, numNodes):
        nsNodes = ns.network.NodeContainer()
        nsNodes.Create(numNodes)
        return nsNodes

    def createInterface(self, Nodes, mobility=True, loss=False, PowerStart=1, PowerEnd=1, Freq=5.180e9):
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

        Devices = wifi.Install(Phy, Mac, Nodes)

        # Mobility for devices
        nsMobility = ns.mobility.MobilityHelper()
        if mobility:
            nsMobility.SetPositionAllocator("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0),
                                          "MinY", ns.core.DoubleValue(0.0), "DeltaX", ns.core.DoubleValue(5.0),
                                          "DeltaY", ns.core.DoubleValue(10.0),
                                          "GridWidth", ns.core.UintegerValue(3), "LayoutType",
                                          ns.core.StringValue("RowFirst"))

        else:
            nsMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        nsMobility.Install(Nodes)

        # Internet for devices
        stack = ns.internet.InternetStackHelper()
        stack.Install(Nodes)

        # IPv4 address for devices
        address = ns.internet.Ipv4AddressHelper()
        address.SetBase(
            ns.network.Ipv4Address("10.1.1.0"),
            ns.network.Ipv4Mask("255.255.255.0")
        )
        Interface = address.Assign(Devices)

        return Interface

    def act_as_server(self, nodes, Interfaces, server_identity, client_identity):
        """current node connection is configured as server, attached to this round's client
             and return ns3 socket for all communication"""
        socket = ns.network.Socket.CreateSocket(
            nodes.Get(server_identity),
            ns.core.TypeId.LookupByName("ns3::UdpSocketFactory"))
        sinkAddress = ns.network.InetSocketAddress(Interfaces.GetAddress(client_identity), 9)
        socket.Connect(sinkAddress)
        return socket

    def act_as_client(self, nodes, client_identity):
        """current node connection is configured as client and return ns3 socket for all communication"""
        socket = ns.network.Socket.CreateSocket(
            nodes.Get(client_identity),
            ns.core.TypeId.LookupByName("ns3::UdpSocketFactory"))
        address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)
        socket.Bind(address)
        return socket

    def make_packets(self, data):
        """pickles data and divides into packets for sending over network"""
        data = str(pickle.dumps(data))[2:-1]
        packetSize = 1024
        dataSize = sys.getsizeof(data)
        numPackets = int(dataSize / packetSize)
        nsPacket = ns.network.Packet(data, dataSize)

        allPackets = []
        # allPackets = [ns.network.Packet(str(pickle.dumps(numPackets + 1))[2:-1], dataSize)]
        for i, start in enumerate(range(0, dataSize, packetSize)):
            if i < numPackets:
                allPackets.append(nsPacket.CreateFragment(start, packetSize))
        allPackets.append(nsPacket.CreateFragment(start, dataSize - start))

        return allPackets

    def send_packet(self, socket, packet_content):
        print("Sending packet content ",ns.core.Simulator.Now().GetSeconds())
        socket.Send(packet_content)

    def send_data(self, socket, obj, attime=0.0):
        """pickle and send data via ns3 socket"""
        packetData = self.make_packets(obj)
        packetSize = 1024
        DataRate = ns.network.DataRate("512kb/s")
        for pkt in packetData:
            ns.core.Simulator.Schedule(ns.core.Seconds(attime), self.send_packet, socket, pkt)
            attime += packetSize * 8.0 / float(DataRate.GetBitRate())

    def getRecvData(self, RecvData):
        for packets in RecvData[1:]:
            RecvData[0].AddAtEnd(packets)
        RecvData = RecvData[0].GetString()
        rcvd_obj = pickle.loads("".join(RecvData).encode().decode("unicode_escape").encode("raw_unicode_escape"))
        return rcvd_obj

    def receive_packets(self, socket):
        print("Receiving packet content ",ns.core.Simulator.Now().GetSeconds())
        packetSize = 1024
        allPackets = []
        numRecvPacket = 0

        tmp = socket.Recv(maxSize=packetSize, flags=0)
        allPackets.append(tmp)
        numRecvPacket += 1
        if numRecvPacket == 1:  # HARDCODED
            self.rcvd_data = self.getRecvData(allPackets)
        # print(self.rcvd_data)

    def recv_data(self, socket):
        """receive data and reverse pickle operation to get object"""
        socket.SetRecvCallback(self.receive_packets)
        # print(self.rcvd_data)
        return self.rcvd_data

    def pass_data(self):
        print("Reading data ", ns.core.Simulator.Now().GetSeconds())
        # time.sleep(1)

    def read_recvd_data(self, attime):
        attime += 1024 * 8.0 / float(512)
        ns.core.Simulator.Schedule(ns.core.Seconds(attime), self.pass_data)
        return self.rcvd_data

    def print_stats(self, os, st):
        print("  Tx Bytes: ", st.txBytes, file=os)
        print("  Rx Bytes: ", st.rxBytes, file=os)
        print("  Tx Packets: ", st.txPackets, file=os)
        print("  Rx Packets: ", st.rxPackets, file=os)
        print("  Lost Packets: ", st.lostPackets, file=os)
        if st.rxPackets > 0:
            print("  Mean{Delay}: ", (st.delaySum.GetSeconds() / st.rxPackets), file=os)
            #print("  Mean{Jitter}: ", (st.jitterSum.GetSeconds() / (st.rxPackets - 1)), file=os)
            print("  Mean{Hop Count}: ", float(st.timesForwarded) / st.rxPackets + 1, file=os)

        for reason, drops in enumerate(st.packetsDropped):
            print("  Packets dropped by reason %i: %i" % (reason, drops), file=os)
        # for reason, drops in enumerate(st.bytesDropped):
        #    print "Bytes dropped by reason %i: %i" % (reason, drops)

    def monitoring_start(self, eta):
        # flow monitoring statistics
        flowmon_helper = ns.flow_monitor.FlowMonitorHelper()
        flowmonitor = flowmon_helper.InstallAll()
        ns.core.Simulator.Stop(ns.core.Seconds(eta))  # FlowMonitor will not work without this
        return flowmon_helper, flowmonitor

    def monitoring_end(self, flowmon_helper, flowmonitor):
        classifier = flowmon_helper.GetClassifier()
        for flow_id, flow_stats in flowmonitor.GetFlowStats():
            t = classifier.FindFlow(flow_id)
            proto = {6: 'TCP', 17: 'UDP'}[t.protocol]
            print("FlowID: %i (%s %s/%s --> %s/%i)" % (
            flow_id, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
            print_stats(sys.stdout, flow_stats)

    def simulation_start(self):
        ns.core.Simulator.Run()

    def simulation_end(self):
        ns.core.Simulator.Destroy()

