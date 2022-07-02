### Command to build base image :
    sudo docker build -t ns3base -f Dockerfile_base_image .

### Command to build application image :
    sudo docker build -t ns3wifi .

### Command to run container and attach to terminal :
    docker run -it ns3wifi /bin/bash