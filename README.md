# Peer-to-Peer Federated Learning Simulator

## PeerFL : A Simulator for P2P FL at scale

Model training on Python (using Keras/TF)   
Network simulation using ns3   
Device constraints like available RAM, GPU applied   

Simulation1 : CIFAR dataset across 10 clients
Simulation 2 : Heterogenous data split and training across 200 clients

## Installation
- Run the ``` run setup.sh ``` file to setup everything.
- If gpu is not required then set the gpu option in ``` config.yml ``` as False.
- If gpu is required then first set the gpu option in ``` config.yml ``` as True and then also run the ``` gpu_setup.sh ``` file.
 

## Running ns3 code
Check that all the above installation steps are completed before running main.py.

Everything can be executed form the ``` main.py ```. Requires passing ns3 home path through command line input. All configurations maintained in ``` config.yml ```. 

Code associated with paper [https://arxiv.org/abs/2405.17839]
