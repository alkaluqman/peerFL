import argparse
import yaml

def func(ops):
    n = ops["n"]
    total_num_of_peers = n

    output_file = open("../../Device/docker-compose.yml", "w")
    output_file.write("version: '2'\n")
    output_file.write("services:\n")
    for i in range(1, total_num_of_peers+1):
        output_file.write("    node"+str(i)+":\n")
        output_file.write("        build: ./peer/\n")
        output_file.write("        environment:\n")
        output_file.write("            - PYTHONUNBUFFERED=1\n")
        output_file.write("            - ORIGIN=node"+str(i)+"\n")
        output_file.write(f'            - PEERS={ops["nodes"][i-1]}\n')
        output_file.write("        volumes:\n")
        output_file.write("            - ./all_data/saved_data_client_"+str(i)+":/usr/thisdocker/dataset:rw\n")
        output_file.write("        privileged: true\n")
        output_file.write("        ports:\n")
        output_file.write("            - '5555'\n")
        output_file.write("        network_mode: none\n")
        output_file.write("        stdin_open: true\n")
        output_file.write("        tty: true\n")
        
    output_file.close()


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/sasuke/repos/p2pFLsim/config.yml", type = str)
    args = parser.parse_args()
    ops = yaml.safe_load(open(args.path, "r"))
    func(ops)
