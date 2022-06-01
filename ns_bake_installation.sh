#!/bin/bash

sudo apt-get update
sudo apt-get install git mercurial
sudo apt-get install g++ 
sudo apt-get install make cmake
sudo apt-get install python3 python3-dev pkg-config sqlite3
sudo apt-get install python3-setuptools
sudo apt-get install gir1.2-goocanvas-2.0 python-gi python-gi-cairo python3-gi python3-gi-cairo
python3-pygraphviz gir1.2-gtk-3.0 ipython3  
sudo apt-get install qt5-default

git clone https://gitlab.com/nsnam/bake

./bake.py configure -e ns-3.32
./bake.py download
./bake.py build