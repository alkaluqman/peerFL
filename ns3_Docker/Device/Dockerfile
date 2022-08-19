FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
COPY peer_ns3.py ./peer.py
COPY ns_helper.py ./ns_helper.py
COPY training.py ./training.py
COPY inference.py ./inference.py
COPY saved_data_test /usr/thisdocker/testset
COPY comm_template.json ./comm_template.json

RUN pip install pyzmq
RUN apt-get update
RUN apt-get install -y iputils-ping
RUN apt-get -y install net-tools
RUN apt-get install -y iperf

EXPOSE 5555-5565