import time
import zmq
import pickle, zlib
import os

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
origin = "server"
socket.setsockopt_string(zmq.IDENTITY, origin)
socket.bind("tcp://*:5555") #from req-rep
# socket.bind("127.0.0.1:5555")
print("[SERVER] Server waiting for connection")

def send_updated_model(socket):
    p = pickle.dumps(global_wt, pickle.HIGHEST_PROTOCOL)
    z = zlib.compress(p)
    socket.send(z, flags=0)
    message = socket.recv()
    print("[CLIENT] Received message : %s" % (message))

#model params
global_wt = {"updated":"model"}
ack = {"hold your":"horses"}
total_clients = 2
models_received_from_clients = 0

while True:
    #  Wait for next request from client
    zi = socket.recv(0) #Reads identity
    z = socket.recv_pyobj(0) #Reads model object
    print("[SERVER] Received object")
    print(zi.decode("utf-8"),"***", z)
    models_received_from_clients +=1
    if models_received_from_clients >= total_clients:
        #sending global model
        print("[SERVER] Sending global model object")
        socket.send_pyobj(global_wt, flags=0)
