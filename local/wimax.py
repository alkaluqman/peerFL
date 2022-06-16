import ns.core
import ns.internet
import ns.applications
import ns.network
import ns.wimax
import ns.mobility

nbss = 10
duration = 7

ssNodes = ns.network.NodeContainer()
bsNodes = ns.network.NodeContainer()

ssNodes.Create(nbss)
bsNodes.Create(1)

scheduler = ns.wimax.WimaxHelper.SCHED_TYPE_SIMPLE   

wimax = ns.wimax.WimaxHelper()
channel = ns.wimax.SimpleOfdmWimaxChannel()
channel.SetPropagationModel(0)

ssDevs = wimax.Install(
    ssNodes,
    wimax.DEVICE_TYPE_SUBSCRIBER_STATION,
    wimax.SIMPLE_PHY_TYPE_OFDM,
    channel,
    scheduler
)

bsDev = wimax.Install(
    bsNodes,
    wimax.DEVICE_TYPE_BASE_STATION,
    wimax.SIMPLE_PHY_TYPE_OFDM,
    channel,
    scheduler
)

bsPos = ns.mobility.ConstantPositionMobilityModel()
bsPos.SetPosition(ns.core.Vector(1000, 0, 0))
bsNodes.Get(0).AggregateObject(bsPos)

#wimax.EnableLogComponents()
ss = [None]*nbss
for i in range(nbss):
    ssPos = ns.mobility.RandomWaypointMobilityModel()
    ssAlloc = ns.mobility.RandomRectanglePositionAllocator()
    xVar = ns.core.UniformRandomVariable()
    xVar.SetAttribute("Min", ns.core.DoubleValue ((i/40.0)*2000))
    xVar.SetAttribute("Max", ns.core.DoubleValue ((i/40.0 + 1)*2000))
    ssAlloc.SetX(xVar)
    yVar = ns.core.UniformRandomVariable()
    yVar.SetAttribute("Min", ns.core.DoubleValue ((i/40.0)*2000))
    yVar.SetAttribute("Max", ns.core.DoubleValue ((i/40.0 + 1)*2000))
    ssAlloc.SetX(yVar)
    ssPos.SetAttribute("PositionAllocator", ns.core.PointerValue(ssAlloc))
    ssPos.SetAttribute("Speed", ns.core.StringValue("ns3::UniformRandomVariable[Min=10.3|Max=40.7]"))
    ssPos.SetAttribute("Pause", ns.core.StringValue("ns3::ConstantRandomVariable[Constant=0.01]"))
    ss[i] = ns.wimax.SubscriberStationNetDevice.GetObject(ssDevs.Get(i))
    ss[i].SetModulationType(ns.wimax.WimaxPhy.MODULATION_TYPE_QAM16_12)
    ssNodes.Get(i).AggregateObject(ssPos)

bs = ns.wimax.BaseStationNetDevice.GetObject(bsDev.Get(0))

#Install
mobility = ns.mobility.MobilityHelper()
mobility.Install(bsNodes)
mobility.Install(ssNodes)

stack = ns.internet.InternetStackHelper()
stack.Install(bsNodes)
stack.Install(ssNodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))

ssInterface = address.Assign(ssDevs)
bsInterface = address.Assign(bsDev)


source = ns.network.Socket.CreateSocket(
    ssNodes.Get(0),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sink = ns.network.Socket.CreateSocket(
    ssNodes.Get(1),
    ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
)

sinkAddress = ns.network.InetSocketAddress(ssInterface.GetAddress(1), 9)
anyAddress = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), 9)

sink.Bind(anyAddress)
source.Connect(sinkAddress)

DlClassifier = ns.wimax.IpcsClassifierRecord(
    ns.network.Ipv4Address("0.0.0.0"),
    ns.network.Ipv4Mask("0.0.0.0"),
    ssInterface.GetAddress(1),
    ns.network.Ipv4Mask("255.255.255.255"),
    0,
    65000,
    9,
    9,
    17,
    1
)

UlClassifier = ns.wimax.IpcsClassifierRecord(
    ssInterface.GetAddress(0),
    ns.network.Ipv4Mask("255.255.255.255"),
    ns.network.Ipv4Address("0.0.0.0"),
    ns.network.Ipv4Mask("0.0.0.0"),
    0,
    65000,
    9,
    9,
    17,
    1
)

UlService = wimax.CreateServiceFlow(
    ns.wimax.ServiceFlow.SF_DIRECTION_UP,
    ns.wimax.ServiceFlow.SF_TYPE_BE,
    UlClassifier
)

DlService = wimax.CreateServiceFlow(
    ns.wimax.ServiceFlow.SF_DIRECTION_DOWN,
    ns.wimax.ServiceFlow.SF_TYPE_BE,
    DlClassifier
)


ss[0].AddServiceFlow(UlService)
ss[1].AddServiceFlow(DlService)




ns.core.Simulator.Stop(ns.core.Seconds(10))
ns.core.Simulator.Run()

ns.core.Simulator.Destroy()
