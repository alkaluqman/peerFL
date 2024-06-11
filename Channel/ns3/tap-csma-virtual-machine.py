# -*-  Mode: Python; -*-
#
# Copyright 2010 University of Washington
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

import sys

import ns.core
import ns.csma
import ns.internet
import ns.network
import ns.tap_bridge


cmd = ns.core.CommandLine()
cmd.numNodes = 2
cmd.totalTime = 1000
cmd.baseName = "Node"
cmd.payLoad = 100000

cmd.AddValue("numNodes", "Number of nodes/devices")
cmd.AddValue("totalTime", "Total simulation time")
cmd.AddValue("baseName", "Base Name of the Node")
cmd.AddValue("payLoad", "payload of the transport layer")

cmd.Parse(sys.argv)

numNodes = int(cmd.numNodes)
totalTime = int(cmd.totalTime)
baseName = cmd.baseName
payLoad = int(cmd.payLoad)


#
# We are interacting with the outside, real, world.  This means we have to 
# interact in real-time and therefore we have to use the real-time simulator
# and take the time to calculate checksums.
#
ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::RealtimeSimulatorImpl"))
ns.core.GlobalValue.Bind("ChecksumEnabled", ns.core.BooleanValue("true"))

#
# Create two ghost nodes.  The first will represent the virtual machine host
# on the left side of the network; and the second will represent the VM on 
# the right side.
#
nodes = ns.network.NodeContainer()
nodes.Create (2)

#
# Use a CsmaHelper to get a CSMA channel created, and the needed net 
# devices installed on both of the nodes.  The data rate and delay for the
# channel can be set through the command-line parser.
#
csma = ns.csma.CsmaHelper()
csma.SetChannelAttribute("DataRate", ns.core.StringValue ("1Mbps"))
devices = csma.Install(nodes)

stack = ns.internet.InternetStackHelper()
stack.Install(nodes)
address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.12.0.0"), ns.network.Ipv4Mask("255.255.255.0"))
Interfaces = address.Assign(devices)
ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

#
# Use the TapBridgeHelper to connect to the pre-configured tap devices for 
# the left side.  We go with "UseLocal" mode since the wifi devices do not
# support promiscuous mode (because of their natures0.  This is a special
# case mode that allows us to extend a linux bridge into ns-3 IFF we will
# only see traffic from one other device on that bridge.  That is the case
# for this configuration.
#
tapBridge = ns.tap_bridge.TapBridgeHelper()
tapBridge.SetAttribute ("Mode", ns.core.StringValue ("UseLocal"))

#
# Connect the right side tap to the right side wifi device on the right-side
# ghost node.
#
for i in range(numNodes):
    tapBridge.SetAttribute ("DeviceName", ns.core.StringValue ("tap-" + baseName + str(i + 1)))
    tapBridge.Install (nodes.Get (i), devices.Get (i))

#
# Run the simulation for ten minutes to give the user time to play around
#
ns.core.Simulator.Stop (ns.core.Seconds (6000))
ns.core.Simulator.Run(signal_check_frequency = -1)
ns.core.Simulator.Destroy()

