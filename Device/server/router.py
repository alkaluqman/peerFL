import time
import zmq
import pickle, zlib
import os
import inference

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

def send_updated_model(socket, obj, identity, protocol=pickle.HIGHEST_PROTOCOL):
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send_multipart([identity, z])

origin = os.environ["ORIGIN"] #"server"
log_prefix = "["+str(origin).upper()+"] "

#Socket creation
context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.setsockopt_string(zmq.IDENTITY, origin)
socket.bind("tcp://*:5555") #from req-rep
print("%sServer waiting for connection" % log_prefix)

#model params
global_wt = {"updated":"model"}
ack = {"hold your":"horses"}
total_clients = int(os.environ["NUM_CLIENTS"])
models_received_from_clients = 0
client_list = []
client_dict = {}

while True:
    #  Wait for next request from client
    client_identity = socket.recv(0) #Reads identity
    client_model = recv_zipped_pickle(socket) #Reads model object
    print("%sReceived object" % log_prefix)
    print(client_identity.decode("utf-8"),"***", client_model)
    client_list.append(client_identity)
    client_dict[client_identity] = client_model
    models_received_from_clients +=1 #TBD: replace with check if model received per identity using dictionary
    if models_received_from_clients >= total_clients:
        # FedAvg to get global model
        global_wt = inference.FedAvg(client_dict)

        # Sending global model
        for i in client_list:
            print("%sSending global model object to %s" % (log_prefix, i.decode("utf-8")))
            # socket.send_multipart([i, b"yoooooo"])
            send_updated_model(socket, global_wt, i)
        models_received_from_clients = 0 #reset for next epoch