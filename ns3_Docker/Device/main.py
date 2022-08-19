import subprocess
import os
import yaml
from argparse import ArgumentParser
import time
import json

def create(numNodes, names, baseName):
    #curr_dir = os.getcwd()

    subprocess.call(" sudo docker-compose up -d", shell=True)

    print("#####################################################")
    print("Docker Container Started")
    print("#####################################################")

    for i in range(numNodes):
        subprocess.call("sudo bash ./setup.sh %s" % names[i], shell = True)

    print("#####################################################")

    print("Creating bridges and tap devices")

    print("#####################################################")
    subprocess.call("sudo bash ./END.sh", shell = True)

    print("#####################################################")

    print("Setup Completed")

    print("#####################################################")
    d_names = [f"device_node{i}_1" for i in range(1, numNodes + 1)]

    #for i in range(numNodes):
    #    subprocess.call(
    #        "sudo docker run --privileged -dit --net=none -v /home/sasuke/repos/p2pFLsim/ns3_Docker/Tests/all_data/saved_data_client_%s:/usr/thisdocker/dataset:rw --name %s %s" % (str(i + 1), d_names[i], "base"), shell = True
    #    )

    #print("#####################################################")

    #print("Docker Container Started")

    #print("#####################################################")
    for i in range(numNodes):
        subprocess.call("sudo bash ./container.sh %s %s % s" % (names[i], i, d_names[i]), shell = True)


    print("#####################################################")

    print("Done")

    print("#####################################################")

    return
    

def ns3(numNodes, baseName):
    totalTime = (10 * 60) * numNodes

    print("######################################################")

    print("Starting ns3 network")

    print("######################################################")

    subprocess.Popen(
        "cd /home/sasuke/repos/bake/source/ns-3.32/ && sudo ./waf --pyrun \"/home/sasuke/repos/p2pFLsim/ns3_Docker/Tests/tap-wifi-virtual-machine.py --numNodes=%s --totalTime=%s --baseName=%s\"" % (str(numNodes), str(totalTime), baseName)
        ,shell = True
        )

    print("Finished Running ns3 simulator")

    return
    


def emulate(numNodes, lastNode):
    print("#####################################################")
    print("Starting Simulation")
    print("#####################################################")
    d_names = [f"device_node{i}_1" for i in range(1, numNodes + 1)]

    for i in range(numNodes):
        if i + 1 != lastNode:
            subprocess.call(
                f"sudo docker exec -d {d_names[i]} python peer.py", shell=True
            )
    subprocess.call(
        f"sudo docker exec {d_names[lastNode-1]} python peer.py", shell=True
    )
        
    #print(process.stdout)
    return 

    
    

def destroy(numNodes, names):
    print("#####################################################")

    print("Destroying Everything")

    print("#####################################################")

    print("#####################################################")

    print("Destroying Docker Containers")

    print("#####################################################")
    d_names = [f"device_node{i}_1" for i in range(1, numNodes + 1)]
    for i in range(numNodes):
        subprocess.call(
            "sudo docker stop %s && sudo docker rm %s" % (d_names[i], d_names[i]), shell=True
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
    parser.add_argument("-bn", "--basename", type = str, action="store", default="Node")
    args = parser.parse_args()

    numNodes = args.number
    emuTime = args.time
    names = []
    baseName = args.basename

    comm_template = json.load(open('/home/sasuke/repos/p2pFLsim/Device/peer/comm_template.json'))
    
    lastNode = int(comm_template[list(*comm_template.keys())[-1]]["to"])
    for i in range(0, numNodes):
        names.append(baseName + str(i + 1))

    operation = args.operation
    if operation == "create":
        create(numNodes, names, baseName)
    elif operation == "ns3":
        ns3(numNodes, baseName)
    elif operation == "emulate":
        emulate(numNodes, lastNode)
    elif operation == "destroy":
        destroy(numNodes, names)
    else:
        raise KeyError

if __name__ ==  '__main__':
    main()


    
