
total_num_of_peers = 7

output_file = open("../docker-compose.yml", "w")
output_file.write("version: '2'\n")
output_file.write("services:\n")
for i in range(1, total_num_of_peers+1):
    output_file.write("    node"+str(i)+":\n")
    output_file.write("        build: ./peer/\n")
    output_file.write("        environment:\n")
    output_file.write("            - PYTHONUNBUFFERED=1\n")
    output_file.write("            - ORIGIN=node"+str(i)+"\n")
    output_file.write("            - PEERS=[]\n")
    output_file.write("        volumes:\n")
    output_file.write("            - ./all_data/saved_data_client_"+str(i)+":/usr/thisdocker/dataset:rw\n")
output_file.close()