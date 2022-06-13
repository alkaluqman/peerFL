import ns.core
import ns.network
import ns.wimax
import ns.internet
import ns.applications

bsNodes = ns.network.NodeContainer()
bsNodes.Create(1)

ssNodes = ns.network.NodeContainer()
ssNodes.Create(2)

wimax = ns.wimax.WimaxHelper()

scheduler = ns.wimax.WimaxHelper.SCHED_TYPE_SIMPLE

ssDevs = wimax.Install(
    ssNodes, ns.wimax.WimaxHelper.DEVICE_TYPE_SUBSCRIBER_STATION
    ,ns.wimax.WimaxHelper.SIMPLE_PHY_TYPE_OFDM,scheduler
)
bsDevs = wimax.Install (bsNodes, ns.wimax.WimaxHelper.DEVICE_TYPE_BASE_STATION,
    ns.wimax.WimaxHelper.SIMPLE_PHY_TYPE_OFDM, scheduler
)
ss = []
for i in range(2):
    tmp = ns.wimax.SubscriberStationNetDevice.GetObject(ssDevs.Get(i))
    tmp.SetModulationType(ns.wimax.WimaxPhy.MODULATION_TYPE_QAM16_12)
    ss.append(tmp)
bs = ns.wimax.BaseStationNetDevice.GetObject(bsDevs.Get(0))

stack = ns.internet.InternetStackHelper()
stack.Install(bsNodes)
stack.Install(ssNodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
ssInterfaces = address.Assign(ssDevs)
bsInterfaces = address.Assign(bsDevs)

wimax.EnableLogComponents()


Server = ns.applications.UdpServerHelper(100)

serverApps = Server.Install(ns.network.NodeContainer(ssNodes.Get(0)))
serverApps.Start(ns.core.Seconds(6.0))
serverApps.Stop(ns.core.Seconds(7))

echoClient = ns.applications.UdpClientHelper(ssInterfaces.GetAddress(0), 100)
echoClient.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
echoClient.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds (1.0)))
echoClient.SetAttribute("PacketSize", ns.core.UintegerValue(1024))
         
clientApps = echoClient.Install(ns.network.NodeContainer(ssNodes.Get(1))) # node5
clientApps.Start(ns.core.Seconds(6.0))
clientApps.Stop(ns.core.Seconds(7))

ns.core.Simulator.Stop(ns.core.Seconds(7.1))

DlClassifierUgs = ns.wimax.IpcsClassifierRecord(
    ssInterfaces.GetAddress(0), ns.network.Ipv4Mask("255.255.255.0"), ns.network.Ipv4Address("0.0.0.0"), ns.network.Ipv4Mask("0.0.0.0"),
    0,65000,
    100,100,17,1
)

DlServiceFlowUgs = wimax.CreateServiceFlow(
    ns.wimax.ServiceFlow.SF_DIRECTION_UP,
    ns.wimax.ServiceFlow.SF_TYPE_RTPS,DlClassifierUgs
)

UlClassifierUgs = ns.wimax.IpcsClassifierRecord(
    ssInterfaces.GetAddress(1), ns.network.Ipv4Mask("255.255.255.0"), ns.network.Ipv4Address("0.0.0.0"), ns.network.Ipv4Mask("0.0.0.0"),
    0,65000,
    100,100,17,1
)

UlServiceFlowUgs = wimax.CreateServiceFlow(
    ns.wimax.ServiceFlow.SF_DIRECTION_UP,
    ns.wimax.ServiceFlow.SF_TYPE_RTPS,UlClassifierUgs
)

ss[0].AddServiceFlow(DlServiceFlowUgs)
ss[1].AddServiceFlow(UlServiceFlowUgs)
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()

