FROM ns3base:latest

#COPY ns_helper.py /pyscripts/ns_helper.py
COPY ns_helper_new.py /pyscripts/ns_helper_new.py
COPY zmq_helper.py /pyscripts/zmq_helper.py
#COPY debugging.py /pyscripts/debugging.py
COPY wifisim.py /pyscripts/wifisim.py
#COPY peer.py /pyscripts/peer.py

WORKDIR /ns/bake/source/ns-3.32/
EXPOSE 9
CMD ["./waf", "--pyrun", "/pyscripts/wifisim.py"]