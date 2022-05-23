import pickle, zlib
import zmq
import os
import random, time

context = zmq.Context()

#  Socket to talk to server
print("[CLIENT] Connecting to hello world server…")
socket = context.socket(zmq.DEALER)
# zmq_address = "tcp://server:5555" #os.environ["ZMQ_ADDRESS"]
# socket.connect(zmq_address)
socket.connect("tcp://localhost:5556")

#model params
local_wt = {"from": "client1"}
ack = {"updated":"client1"}

# 1 trainig epoch
time.sleep(1)

print("[CLIENT] Sending iteration %s …" % (str(local_wt)))
p = pickle.dumps(local_wt, pickle.HIGHEST_PROTOCOL)
z = zlib.compress(p)
socket.send(z, flags=0)
# message = socket.recv()
# print("[CLIENT] Received message : %s" % (message))

#wait for global model
# while True:
z = socket.recv(0)
if z:
    p = zlib.decompress(z)
    message = pickle.loads(p)
    print("[CLIENT] Received updated global model")
    print(message)

        #  Send acknowledgement back to server since REQ-REP sockets MUST alternate calls
        # socket.send_string(str(ack))