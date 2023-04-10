from posixpath import basename
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
from argparse import ArgumentParser
import time
import json
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
import shutil
import sys

#, "2": {"from": "2", "to": "3"}, "3": {"from": "3", "to": "4"}}
def create(numNodes, names, baseName):
    curr_dir = os.getcwd()

    print("#####################################################")
    print("Generating Data")
    print("#####################################################")

    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(32,32,3), classes=10)
    base_model.save("./base_model")    

    subprocess.call(f"cd ./Device/ && python data_for_docker.py -n {numNodes} && mv ./all_data/saved_data_test ./peer/saved_data_test ", shell=True, stdout=subprocess.DEVNULL)

    for i in range(1, numNodes+1):
        name = f"./Device/all_data/saved_data_client_{i}"
        subprocess.call(f"cp -r ./base_model {name}", shell=True)

    subprocess.call("cd ./Device/ && sudo docker-compose up -d", shell=True)

    print("#####################################################")
    print("Docker Container Started")
    print("#####################################################")

    for i in range(numNodes):
        subprocess.call("cd ./setup && sudo bash ./setup.sh %s" % names[i], shell = True)
        print(names[i])

    print("#####################################################")

    print("Creating bridges and tap devices")

    print("#####################################################")
    subprocess.call("cd ./setup && sudo bash ./END.sh", shell = True)

    print("#####################################################")

    print("Setup Completed")

    print("#####################################################")
    d_names = [f"device_node{i}_1" for i in range(1, numNodes + 1)]

    for i in range(numNodes):
        subprocess.call("cd ./setup && sudo bash ./container.sh %s %s % s" % (names[i], i, d_names[i]), shell = True)

    print("#####################################################")

    print("Done")

    print("#####################################################")

    return
    

def ns3(numNodes, baseName, ns3Path):
    totalTime = (100 * 60) * numNodes

    print("######################################################")

    print("Starting ns3 network")

    print("######################################################")

    subprocess.Popen(
        f"cd {ns3Path} && sudo ./waf --pyrun \"%s --numNodes=%s --totalTime=%s --baseName=%s\"" % (os.path.join(os.getcwd(), "ns3/tap-wifi-virtual-machine.py"), str(numNodes), str(totalTime), baseName)
        ,shell = True
        )

    print("Ns3 simulator is running")

    return
    


def emulate(numNodes, lastNode, central=False, server=None):
    print("#####################################################")
    print("Starting Simulation")
    print("#####################################################")

    if central:
        d_names = [f"device_node{i}_1" for i in range(1, numNodes + 1)]

        start_time = time.time()
        for i in range(1, numNodes):
                subprocess.call(
                    f"sudo docker exec -d {d_names[i]} python peer.py", shell=True
                )
        
        subprocess.call(
            f"sudo docker exec {d_names[0]} python peer.py", shell=True
        )
        tot_time = time.time() - start_time
        print(f"TOTAL TIME TAKEN: {tot_time}")
    
    else:
        d_names = [f"device_node{i}_1" for i in range(1, numNodes + 1)]

        start_time = time.time()
        for i in range(numNodes):
            if i + 1 != lastNode:
                subprocess.call(
                    f"sudo docker exec -d {d_names[i]} sudo bash ./network_config.sh", shell=True
                )
        
        subprocess.call(
            f"sudo docker exec {d_names[lastNode-1]} sudo bash ./network_config.sh", shell=True
        )
        tot_time = time.time() - start_time
        print(f"TOTAL TIME TAKEN: {tot_time}")
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

    print("Destroying Docker Images")

    print("#####################################################")

    subprocess.call(
        "sudo docker image rm $(sudo docker image ls --format '{{.Repository}}:{{.Tag}}' | grep 'device_node*')", shell=True
    )

    print("#####################################################")

    print("Destroying Docker Bridges")

    print("#####################################################")
    
    for i in range(numNodes):
        subprocess.call(
            "cd ./setup && sudo bash ./destroy.sh %s" % (names[i]), shell = True
        )

    print("#####################################################")

    print("Removing Local Data")

    print("#####################################################")

    subprocess.call("sudo rm -rf ./base_model", shell=True)
    subprocess.call("cd ./Device && sudo rm -rf ./all_data && sudo rm -rf ./peer/saved_data_test", shell=True)

    print("#####################################################")

    print("Done")

    print("#####################################################")

    return


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--time", type = int, action = "store", default = 10)
    parser.add_argument("-op", "--operation", type = str, required=True)
    parser.add_argument("-bn", "--baseName", type = str, action="store", default="Node")
    parser.add_argument("-p", "--path", type=str, default="config.yml", help="No need to chnage this by default.")
    parser.add_argument("-nsp", "--ns3Path", type=str, default="/home/sasuke/repos/bake/source/ns-3.32/", help="Ns3 home directory path.")
    args = parser.parse_args()


    emuTime = args.time
    names = []
    baseName = args.baseName
    ops = yaml.safe_load(open(args.path, "r"))
    central = ops['central']
    numNodes = ops['n']
    
    path = os.path.join(os.getcwd(), args.path)
    subprocess.call(f"cd ./setup && python comm_template_helper.py -p {path}", shell = True)
    subprocess.call(f"cd ./setup && python docker_compose_helper.py -p {path}", shell = True)
    
    comm_template = json.load(open('./Device/peer/comm_template.json'))
    lastNode = int(comm_template[list(comm_template.keys())[-1]]["to"])
    for i in range(0, numNodes):
        names.append(baseName + str(i + 1))

    operation = args.operation
    if operation == "create":
        create(numNodes, names, baseName)
    elif operation == "ns3":
        ns3(numNodes, baseName, args.ns3Path)
    elif operation == "emulate":
        emulate(numNodes, lastNode, central)
    elif operation == "destroy":
        destroy(numNodes, names)
    else:
        raise KeyError

if __name__ ==  '__main__':
    shutil.copyfile("./config.yml", "./Device/peer/config.yml")

    main()

    os.remove("./Device/peer/config.yml")


    
