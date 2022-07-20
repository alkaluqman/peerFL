import subprocess
import os
import yaml
from argparse import ArgumentParser
import time

def create(numNodes, names, baseName):

    for i in range(numNodes):
        subprocess.call("sudo bash ./setup.sh %s" % names[i], shell = True)

    print("#####################################################")

    print("Creating bridges and tap devices")

    print("#####################################################")
    subprocess.call("sudo bash ./END.sh", shell = True)

    print("#####################################################")

    print("Setup Completed")

    print("#####################################################")

    for i in range(numNodes):
        subprocess.call(
            "sudo docker run --privileged -dit --net=none --name %s %s" % (names[i], "ns3base"), shell = True
        )

    print("#####################################################")

    print("Docker Container Started")

    print("#####################################################")

    for i in range(numNodes):
        subprocess.call("sudo bash ./container.sh %s %s" % (names[i], i), shell = True)


    print("#####################################################")

    print("Done")

    print("#####################################################")

    return
    

def ns3(numNodes):
    total_emu_time = (5 * 60) * numNodes

    print("######################################################")

    print("Starting ns3 network")

    print("######################################################")

    subprocess.Popen(
        "cd home/sasuke/repos/bake/source/ns-3.32/ && sudo ./waf --pyrun /home/sasuke/repos/p2pFLsim/ns3_Docker/Tests/tap-wifi-virtual-machine.py"
        ,shell = True
        )

    print("Finished Running ns3 simulator")

    return
    


def emulate(numNodes, names):
    #Starting  Sim

    for  i in range(0, numNodes):
        subprocess.call(
            "sudo docker restart -t 0 %s" % names[i], shell = True
        )
    
    for i in range(0, numNodes):
        subprocess.call("sudo bash ./container.sh %s %s" % (names[i], i), shell = True)
    
    return 

    
    

def destroy(numNodes, names):
    print("#####################################################")

    print("Destroying Everything")

    print("#####################################################")

    print("#####################################################")

    print("Destroying Docker Containers")

    print("#####################################################")

    for i in range(numNodes):
        subprocess.call(
            "sudo docker stop %s && sudo docker rm %s" % (names[i], names[i]), shell=True
        )

    print("#####################################################")

    print("Destroying Docker Bridges")

    print("#####################################################")
    
    for i in range(numNodes):
        subprocess.call(
            "sudo bash ./destroy.sh %s" % (names[i]), shell = True
        )

    print("#####################################################")

    print("Done")

    print("#####################################################")

    return


def main():
    parser = ArgumentParser()
    parser.add_argument("-n", "--number", type = int, action = "store", default = 2)
    parser.add_argument("-t", "--time", type = int, action = "store", default = 10)
    parser.add_argument("-op", "--operation", type = str, required=True)
    args = parser.parse_args()

    numNodes = args.number
    emuTime = args.time
    names = []
    baseName = "Node"

    for i in range(0, numNodes):
        names.append(baseName + str(i + 1))

    operation = args.operation
    if operation == "create":
        create(numNodes, names, baseName)
    elif operation == "ns3":
        ns3()
    elif operation == "emulation":
        emulate(numNodes, names)
    elif operation == "destroy":
        destroy(numNodes, names)
    else:
        raise KeyError

if __name__ ==  '__main__':
    main()


    
