FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
# Tensorflow 2.8, min 2.7 required for dataset save, load functions to work

RUN apt-get update 
RUN apt-get install -y git mercurial
RUN apt-get install -y g++ 
RUN apt-get install -y make cmake
RUN apt-get install -y python3 python3-dev pkg-config sqlite3
RUN apt-get install -y python3-setuptools
RUN apt-get install -y gir1.2-goocanvas-2.0 python-gi python-gi-cairo python3-gi python3-gi-cairo python3-pygraphviz gir1.2-gtk-3.0 ipython3  
RUN apt-get install -y qt5-default
RUN apt-get install -y castxml
RUN apt-get install -y iputils-ping

WORKDIR /ns
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN git clone https://gitlab.com/nsnam/bake
WORKDIR /ns/bake
RUN ./bake.py configure -e ns-3.32
RUN ./bake.py download
RUN ./bake.py build

COPY packet.cc /ns/bake/source/ns-3.32/src/network/model/packet.cc
COPY packet.h /ns/bake/source/ns-3.32/src/network/model/packet.h
WORKDIR /ns/bake/source/ns-3.32/
RUN ./waf configure
RUN ./waf --apiscan=all
RUN ./waf build