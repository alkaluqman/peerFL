import pickle, zlib
import zmq
import os
import random, time

context = zmq.Context()

#  Socket to talk to server
print("[CLIENT] Connecting to hello world server…")
socket = context.socket(zmq.DEALER)
origin = "client1"
socket.setsockopt_string(zmq.IDENTITY, origin)
socket.connect("tcp://localhost:5555")

#model params
local_wt = {"from": origin}
ack = {"updated": origin}

# 1 training epoch
time.sleep(1)

print("[CLIENT] Sending iteration %s …" % (str(local_wt)))
socket.send_pyobj(local_wt, flags=0)

#wait for global model
while True:
    zi = socket.recv(0) #Reads identity
    z = socket.recv_pyobj(0) #Reads model object
    print("[SERVER] Received object")
    print(zi.decode("utf-8"),"***", z)