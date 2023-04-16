#!/bin/bash

#Docker Installation
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)"  -o /usr/local/bin/docker-compose
sudo mv /usr/local/bin/docker-compose /usr/bin/docker-compose
sudo chmod +x /usr/bin/docker-compose

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

#Install Python
sudo apt-get install -y python3-pip
sudo apt-get install -y python-is-python3

#NS3 Installation
sudo apt-get update
sudo apt-get install git mercurial
sudo apt-get install g++ 
sudo apt-get install make cmake
sudo apt-get install python3 python3-dev pkg-config sqlite3
sudo apt-get install python3-setuptools
sudo apt-get install -y gir1.2-goocanvas-2.0 python-gi python-gi-cairo python3-gi python3-gi-cairo python3-pygraphviz gir1.2-gtk-3.0 ipython3  
sudo apt-get install -y qt5-default
sudo apt-get install -y castxml
sudo apt-get install -y iputils-ping
sudo apt-get install -y bridge-utils 
sudo apt-get install -y uml-utilities
sudo apt-get install -y net-tools
pip install pygccxml
pip install cxxfilt

git clone https://gitlab.com/nsnam/bake

cd ./bake

./bake.py configure -e ns-3.32
./bake.py download
./bake.py build

cd ./source/ns-3.32/

./waf configure
./waf --apiscan=all
./waf build

#Python requirements
pip install -r requirements.txt

sudo apt-get update

#