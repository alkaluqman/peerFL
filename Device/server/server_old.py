import time
import zmq
import pickle, zlib
import os

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
zmq_address = "tcp://*:5555" #os.environ["ZMQ_ADDRESS"]
socket.bind("tcp://*:5556")
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
    z = socket.recv(0)
    if z:
        p = zlib.decompress(z)
        message = pickle.loads(p)

        print("[SERVER] Received object")
        print(message)
        models_received_from_clients +=1

        #  Send acknowledgement back to client since REQ-REP sockets MUST alternate calls
        # socket.send_string(str(ack))
        # if models_received_from_clients == total_clients:
        send_updated_model(socket)