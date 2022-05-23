import pickle, zlib
import zmq
import os
import random, time
import training

def send_zipped_pickle(socket, obj, flags=0, protocol=pickle.HIGHEST_PROTOCOL):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)


def recv_zipped_pickle(socket, flags=0):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)

origin = os.environ["ORIGIN"] #"client1"
log_prefix = "["+str(origin).upper()+"] "

#  Socket to talk to server. Dealer has to make first contact with Router
print("%sConnecting to server…" % log_prefix)
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt_string(zmq.IDENTITY, origin)
# socket.connect("tcp://localhost:5555")
socket.connect("tcp://server:5555")

#model params
local_wt = {"from": origin}
ack = {"updated": origin}

# Num of training epochs need to be specified at client side
for i in range(0,1):
    #local model training
    local_wt = training.local_training(origin)

    print("%sSending iteration %s …" % (log_prefix, str(i)))
    send_zipped_pickle(socket, local_wt)

    #wait for global model
    # while True:
    z = recv_zipped_pickle(socket)
    print("%sReceived object %s version %s" % (log_prefix, str(z), str(i)))
