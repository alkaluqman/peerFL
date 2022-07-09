FROM ns3base:latest

COPY ns_helper_new.py ./ns_helper_new.py
COPY debugging.py ./peer.py
#COPY zmq_helper.py ./zmq_helper.py
#COPY training.py ./training.py
#COPY inference.py ./inference.py
COPY ./all_data/saved_data_test /usr/thisdocker/testset
COPY comm_template.json ./comm_template.json


WORKDIR /ns/bake/source/ns-3.32/
EXPOSE 9
CMD ["./waf", "--pyrun", "peer.py"]