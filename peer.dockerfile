FROM ubuntu
COPY bake ./bake
COPY packet.cc ./packet.cc
COPY packet.h ./packet.h
COPY peer.py ./peer.py
RUN mv packet.cc bake/source/ns-3.32/src/network/model/
RUN mv packet.h bake/source/ns-3.32/src/network/model/

RUN cd bake/source/ns-3.32/

CMD["./waf --pyrun", "peer.py"]
