/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 MIRKO BANCHI
 * Copyright (c) 2015 University of Washington
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Mirko Banchi <mk.banchi@gmail.com>
 *          Sebastien Deronne <sebastien.deronne@gmail.com>
 *          Tom Henderson <tomhend@u.washington.edu>
 *
 * Adapted from wifi-ht-network.cc example
 */
#include <iostream>
#include <fstream>
#include <iomanip>
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/wifi-mac.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"

#include "ns3/flow-monitor-module.h"
// #include "ns3/dca-txop.h"
#include "ns3/pointer.h"
#include "ns3/on-off-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/qos-txop.h"
#include "ns3/edca-parameter-set.h"
#include "ns3/txop.h"
#include "ns3/applications-module.h"
#include "ns3/rng-seed-manager.h"
#define PI 3.14159265
// This is a simple example of an IEEE 802.11n Wi-Fi network.
//
// The main use case is to enable and test SpectrumWifiPhy vs YansWifiPhy
// under saturation conditions (for max throughput).
//
// Network topology:
//
//  Wi-Fi 192.168.1.0
//
//   STA                  AP
//    * <-- distance -->  *
//    |                   |
//    n1                  n2
//
// Users may vary the following command-line arguments in addition to the
// attributes, global values, and default values typically available:
//
//    --simulationTime:  Simulation time in seconds [10]
//    --distance:        meters separation between nodes [50]
//    --index:           restrict index to single value between 0 and 31 [256]
//    --wifiType:        select ns3::SpectrumWifiPhy or ns3::YansWifiPhy [ns3::SpectrumWifiPhy]
//    --errorModelType:  select ns3::NistErrorRateModel or ns3::YansErrorRateModel [ns3::NistErrorRateModel]
//    --enablePcap:      enable pcap output [false]
//
// By default, the program will step through 64 index values, corresponding
// to the following MCS, channel width, and guard interval combinations:
//   index 0-7:    MCS 0-7, long guard interval, 20 MHz channel
//   index 8-15:   MCS 0-7, short guard interval, 20 MHz channel
//   index 16-23:  MCS 0-7, long guard interval, 40 MHz channel
//   index 24-31:  MCS 0-7, short guard interval, 40 MHz channel
//   index 32-39:    MCS 8-15, long guard interval, 20 MHz channel
//   index 40-47:   MCS 8-15, short guard interval, 20 MHz channel
//   index 48-55:  MCS 8-15, long guard interval, 40 MHz channel
//   index 56-63:  MCS 8-15, short guard interval, 40 MHz channel
// and send packets at a high rate using each MCS, using the SpectrumWifiPhy
// and the NistErrorRateModel, at a distance of 1 meter.  The program outputs
// results such as:
//
// wifiType: ns3::SpectrumWifiPhy distance: 1m
// index   MCS   width Rate (Mb/s) Tput (Mb/s) Received
//     0     0      20       6.5     5.96219    5063
//     1     1      20        13     11.9491   10147
//     2     2      20      19.5     17.9184   15216
//     3     3      20        26     23.9253   20317
//     ...
//
// selection of index values 32-63 will result in MCS selection 8-15
// involving two spatial streams

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiSpectrumSaturationExample");



double * simulate (uint16_t n,  uint32_t minCw[],double SimTime,uint32_t seed)
{
    // double distance = 1;
    double simulationTime = SimTime ; //seconds
    // uint16_t index = 0;
    uint16_t numStaNodes = n;
    // uint32_t channelWidth = 0;
    std::string wifiType = "ns3::YansWifiPhy";
    std::string errorModelType = "ns3::NistErrorRateModel";
    // bool enablePcap = true;

  // CommandLine cmd (__FILE__);
  // cmd.AddValue ("simulationTime", "Simulation time in seconds", simulationTime);
  // cmd.AddValue ("distance", "meters separation between nodes", distance);
  // cmd.AddValue ("index", "restrict index to single value between 0 and 63", index);
  // cmd.AddValue ("wifiType", "select ns3::SpectrumWifiPhy or ns3::YansWifiPhy", wifiType);
  // cmd.AddValue ("errorModelType", "select ns3::NistErrorRateModel or ns3::YansErrorRateModel", errorModelType);
  // cmd.AddValue ("enablePcap", "enable pcap output", enablePcap);
  // cmd.Parse (argc,argv);

    // std::cout << "wifiType: " << wifiType << " distance: " << distance << "m" << std::endl;
    
    // uint32_t payloadSize = 1472;
    // payloadSize = 1472; // 1500 bytes IPv4
      // uint32_t cwmin = 512;
      // Config::SetDefault("ns3::DcaTxop::MinCw", UintegerValue (cwmin));

    NodeContainer wifiStaNode;
    wifiStaNode.Create (numStaNodes);
    NodeContainer wifiApNode;
    wifiApNode.Create (1);

    YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
    SpectrumWifiPhyHelper spectrumPhy = SpectrumWifiPhyHelper::Default ();
    if (wifiType == "ns3::YansWifiPhy")
    {
        YansWifiChannelHelper channel;
        channel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
        channel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
        phy.SetChannel (channel.Create ());
        phy.Set ("TxPowerStart", DoubleValue (1));
        phy.Set ("TxPowerEnd", DoubleValue (1));
    }
    else if (wifiType == "ns3::SpectrumWifiPhy")
    {
        Ptr<MultiModelSpectrumChannel> spectrumChannel
            = CreateObject<MultiModelSpectrumChannel> ();
        Ptr<FriisPropagationLossModel> lossModel
            = CreateObject<FriisPropagationLossModel> ();
        spectrumChannel->AddPropagationLossModel (lossModel);

        Ptr<ConstantSpeedPropagationDelayModel> delayModel
            = CreateObject<ConstantSpeedPropagationDelayModel> ();
        spectrumChannel->SetPropagationDelayModel (delayModel);

        spectrumPhy.SetChannel (spectrumChannel);
        spectrumPhy.SetErrorRateModel (errorModelType);
        spectrumPhy.Set ("Frequency", UintegerValue (5180)); // channel 36 at 20 MHz
        spectrumPhy.Set ("TxPowerStart", DoubleValue (1));
        spectrumPhy.Set ("TxPowerEnd", DoubleValue (1));
      }
    else
    {
        NS_FATAL_ERROR ("Unsupported WiFi type " << wifiType);
    }

    WifiHelper wifi;
    wifi.SetStandard (WIFI_PHY_STANDARD_80211b);
    WifiMacHelper mac;

    Ssid ssid = Ssid ("ns380211b");

      
    /////////////////////////////////////////////////////////////////////////////////////////
    // wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", DataRate,
                                    // "ControlMode", DataRate);
    /////////////////////////////////////////////////////////////////////////////////////////
    StringValue phymode = StringValue ("DsssRate1Mbps");
    wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", phymode,"ControlMode",phymode);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Multi rate wifi
    // std::string wifiManager ("Arf");
    // wifi.SetRemoteStationManager ("ns3::" + wifiManager + "WifiManager");

    NetDeviceContainer staDevices;
    NetDeviceContainer apDevice;

    RngSeedManager::SetSeed(seed);
    if (wifiType == "ns3::YansWifiPhy")
    {
        mac.SetType ("ns3::StaWifiMac",
                     "Ssid", SsidValue (ssid));
        staDevices = wifi.Install (phy, mac, wifiStaNode);
        // mac.SetType ("ns3::StaWifiMac",
        //              "Ssid", SsidValue (ssid));
        // staDeviceB = wifi.Install (phy, mac, wifiStaNode.Get(1));
        mac.SetType ("ns3::ApWifiMac",
                     "Ssid", SsidValue (ssid));
        apDevice = wifi.Install (phy, mac, wifiApNode);

    }
    else if (wifiType == "ns3::SpectrumWifiPhy")
    {
        mac.SetType ("ns3::StaWifiMac",
                     "Ssid", SsidValue (ssid));
        staDevices = wifi.Install (spectrumPhy, mac, wifiStaNode);
        // mac.SetType ("ns3::StaWifiMac",
        //              "Ssid", SsidValue (ssid));
        // staDeviceB = wifi.Install (spectrumPhy, mac, wifiStaNode.Get(1));
        mac.SetType ("ns3::ApWifiMac",
                     "Ssid", SsidValue (ssid));
        apDevice = wifi.Install (spectrumPhy, mac, wifiApNode);
    }

    // Setting cwMin Parameter
    // uint32_t minCw [n] =  { 32, 64, 64, 64, 64 };
    for (uint32_t i = 0; i < numStaNodes; i++)
    {
    // // Ptr<Node> node = wifiStaNode.Get(0); // Get station from node container 
        Ptr<NetDevice> dev = wifiStaNode.Get(i)->GetDevice(0);
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(dev);
        Ptr<WifiMac> mac1 = wifi_dev->GetMac();
        PointerValue ptr;
        mac1->GetAttribute("Txop", ptr);
        Ptr<Txop> dca = ptr.Get<Txop>();
        // std::cout<<dca->GetMinCw()<<std::endl;
        dca->SetMinCw(minCw[i]);
        // std::cout<<dca->GetMinCw()<<std::endl;
    }
    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

    // AP at origin
    positionAlloc->Add (Vector (0.0, 0.0, 0.0));

    //locate N stations as a circle with a center at AP
    //for N stations, set rho as radius and theta
    //then calculate position of each station  
    float rho = 1;
    for (uint32_t i = 0; i < numStaNodes; i++)
    {
      double theta = i * 2 * PI / numStaNodes;
      positionAlloc->Add (Vector (rho * cos(theta), rho * sin(theta), 0.0));
    }
    mobility.SetPositionAllocator (positionAlloc);

    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

    mobility.Install (wifiApNode);
    mobility.Install (wifiStaNode);


    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install (wifiStaNode);
    stack.Install (wifiApNode);

    Ipv4AddressHelper address;
    address.SetBase ("192.168.1.0", "255.255.255.0");

    Ipv4InterfaceContainer staNodeInterfaces;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterfaces = address.Assign (staDevices);
    apNodeInterface = address.Assign (apDevice);

/////////////////////////////////////////////////////////////////////

    ApplicationContainer cbrApps;
    uint16_t cbrPort = 12345;
    OnOffHelper onOffHelper ("ns3::UdpSocketFactory", InetSocketAddress (apNodeInterface.GetAddress(0), cbrPort));
    onOffHelper.SetAttribute ("PacketSize", UintegerValue (1400));
    onOffHelper.SetAttribute ("OnTime",  StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
    onOffHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));



    for (uint32_t i = 0; i < numStaNodes; i++)
    {
        // flow 1:  node 0 -> node 1
        onOffHelper.SetAttribute ("DataRate", StringValue ("1000000bps"));
        onOffHelper.SetAttribute ("StartTime", TimeValue (Seconds (1.000000+(i*0.00001))));
        cbrApps.Add (onOffHelper.Install (wifiStaNode.Get (i)));
    }
  // flow 2:  node 2 -> node 1
  /** \internal
   * The slightly different start times and data rates are a workaround
   * for \bugid{388} and \bugid{912}
   */


 /** \internal
   * We also use separate UDP applications that will send a single
   * packet before the CBR flows start.
   * This is a workaround for the lack of perfect ARP, see \bugid{187}
   */
    uint16_t  echoPort = 9;
    UdpEchoClientHelper echoClientHelper (apNodeInterface.GetAddress(0), echoPort);
    echoClientHelper.SetAttribute ("MaxPackets", UintegerValue (1));
    echoClientHelper.SetAttribute ("Interval", TimeValue (Seconds (0.1)));
    echoClientHelper.SetAttribute ("PacketSize", UintegerValue (10));
    ApplicationContainer pingApps;

    for (uint32_t i = 0; i < numStaNodes; i++)
    {
        // again using different start times to workaround Bug 388 and Bug 912
        echoClientHelper.SetAttribute ("StartTime", TimeValue (Seconds (0.001+(i*0.005))));
        pingApps.Add (echoClientHelper.Install (wifiStaNode.Get (i)));
        // echoClientHelper.SetAttribute ("StartTime", TimeValue (Seconds (0.006)));
        // pingApps.Add (echoClientHelper.Install (wifiStaNode.Get (1)));
    }


    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll ();

    // 9. Run simulation for 10 seconds
    Simulator::Stop (Seconds (simulationTime + 1));
    Simulator::Run ();

    // 10. Print per flow statistics
    monitor->CheckForLostPackets ();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();
    static double obs[3];
    double otherThpt = 0;
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
    {
        // first 2 FlowIds are for ECHO apps, we don't want to display them
        //
        // Duration for throughput measurement is 9.0 seconds, since
        //   StartTime of the OnOffApplication is at about "second 1"
        // and
        //   Simulator::Stops at "second 10".
        uint16_t one = 1;
        if (i->first == (numStaNodes+one))
        {
            // Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
            // std::cout << "Flow " << i->first - numStaNodes << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
            // std::cout << "  Tx Packets: " << i->second.txPackets << "\n";
            // std::cout << "  Tx Bytes:   " << i->second.txBytes << "\n";
            // std::cout << "  TxOffered:  " << i->second.txBytes * 8.0 / simulationTime / 1000 / 1000  << " Mbps\n";
            // std::cout << "  Rx Packets: " << i->second.rxPackets << "\n";
            // std::cout << "  Rx Bytes:   " << i->second.rxBytes << "\n";
            // std::cout << "  Throughput: " << i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000  << " Mbps\n";
            // std::cout << "  Throughput(fraction): " << i->second.rxBytes * 8.0 / simulationTime  / 1000 / 1000 /1<<"\n";
            obs[0] = i->second.txPackets;
            obs[1] = i->second.rxPackets;
        }

        if (i->first > (numStaNodes+one))
        {
            // Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
            // std::cout << "Flow " << i->first - numStaNodes << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
            // std::cout << "  Tx Packets: " << i->second.txPackets << "\n";
            // std::cout << "  Tx Bytes:   " << i->second.txBytes << "\n";
            // std::cout << "  TxOffered:  " << i->second.txBytes * 8.0 / simulationTime / 1000 / 1000  << " Mbps\n";
            // std::cout << "  Rx Packets: " << i->second.rxPackets << "\n";
            // std::cout << "  Rx Bytes:   " << i->second.rxBytes << "\n";
            // std::cout << "  Throughput: " << i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000  << " Mbps\n";
            // std::cout << "  Throughput(fraction): " << i->second.rxBytes * 8.0 / simulationTime  / 1000 / 1000 /1<<"\n";
            otherThpt = otherThpt+i->second.rxPackets; 
        }
        obs[2] = otherThpt;
    }
    // 11. Cleanup
    Simulator::Destroy ();


/////////////////////////////////////////////////////////////////////  







      // /* Setting applications */
      // uint16_t port = 9;
      // UdpServerHelper server (port);

      // ApplicationContainer serverApp = server.Install (wifiStaNode.Get (0));
      // serverApp.Start (Seconds (0.0));
      // serverApp.Stop (Seconds (simulationTime + 1));

      // UdpClientHelper client (staNodeInterface.GetAddress (0), port);
      // client.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
      // client.SetAttribute ("Interval", TimeValue (Time ("0.0001"))); //packets/s
      // client.SetAttribute ("PacketSize", UintegerValue (payloadSize));

      // ApplicationContainer clientApp = client.Install (wifiApNode.Get (0));
      // clientApp.Start (Seconds (1.0));
      // clientApp.Stop (Seconds (simulationTime + 1));


///////////////////////////////////////////////////////////////////////////

      // if (enablePcap)
      //   { std::stringstream ss;
      //     ss << "wifi-spectrum-saturation-example-" << i;
      //     phy.EnablePcap (ss.str (), staDeviceA);
      //   }

      // Simulator::Stop (Seconds (simulationTime + 1));
      // Simulator::Run ();

      // double throughput;
      // uint64_t totalPacketsThrough;
      // totalPacketsThrough = DynamicCast<UdpServer> (serverApp.Get (0))->GetReceived ();
      // throughput = totalPacketsThrough * payloadSize * 8 / (simulationTime * 1000000.0); //Mbit/s
      // std::cout << std::setw (5) << i <<
      //   std::setw (6) << (i % 8) + 8 * (i / 32) <<
      //   std::setw (8) << channelWidth <<
      //   std::setw (10) << datarate <<
      //   std::setw (12) << throughput <<
      //   std::setw (8) << totalPacketsThrough <<
      //   std::endl;
      // Simulator::Destroy ();
    
    return obs;
}


void writeArraytoTxt(const std::string& fileName,double arr[],uint16_t arrayLen)
  {
      std::ofstream out_stream;
      std::cout.precision(4);
      out_stream.open(fileName);
    for (uint32_t i = 0; i < arrayLen; i++)
    {
// 
        // std::cout<<op[i]<<std::endl;
        out_stream << arr[i] << std::endl;
// 
    }
    out_stream.close();
}

int main()
{   

    uint16_t n = 10;
    uint32_t otherCW = 128;
    uint32_t numExample = 500;
    double SimTime = 20.0;

    // Node 1 CW Range define
    // uint32_t cwLow = 32;
    // uint32_t cwHigh = 512;
    // uint32_t cwDiff = 8;

    // uint32_t actionDim = ((cwHigh-cwLow)/cwDiff)+1;
// 
    // uint32_t cw1List[actionDim];
    
    // for (uint32_t l = 0;l<actionDim;l++)
    // {
        // cw1List[l] = cwLow+(l*cwDiff);
    // }
    uint32_t cw1List[9] = {32,48,64,96,128,192,256,384,512};
    uint32_t actionDim = 9;
    // CW initialization
    
    uint32_t minCw [n];
    for (uint16_t k = 0;k<n;k++)
    {
        minCw[k] = otherCW;
    }
   
    double *op;
    double node1Tx[numExample];
    double node1Rx[numExample];
    double otherRx[numExample]; 

    for (uint32_t j = 0;j < actionDim; j++)
    { 
        minCw[0] = cw1List[j];
    for (uint32_t i = 0; i < numExample; i++)
    {
         
        op = simulate(n,minCw,SimTime,i+cw1List[j]);//add some random no
        node1Tx[i] = op[0];
        node1Rx[i] = op[1];
        otherRx[i] = op[2];
    }

    std::string fileNameBase = std::to_string(minCw[0])+'+'+std::to_string(minCw[1]); // 32+32
    std::string fileExt = ".txt";
    fileNameBase = fileNameBase+fileExt; // 32+32.txt

    std::string prefix1 = "./Dataset/10Node/node1TxPackets";
    std::string prefix2 = "./Dataset/10Node/node1RxPackets";
    std::string prefix3 = "./Dataset/10Node/otherRxPackets";

        // std::string fileNameBase2 = "flow2";
    std::string fileName1 = prefix1+'+'+fileNameBase;
    std::string fileName2 = prefix2+'+'+fileNameBase;
    std::string fileName3 = prefix3+'+'+fileNameBase;

    std::cout<<fileName1<<std::endl;
    std::cout<<fileName2<<std::endl;
    std::cout<<fileName3<<std::endl;

    writeArraytoTxt(fileName1,node1Tx,numExample);
    writeArraytoTxt(fileName2,node1Rx,numExample);
    writeArraytoTxt(fileName3,otherRx,numExample);

    }
return 0;

}
