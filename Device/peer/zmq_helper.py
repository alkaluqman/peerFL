import zmq
import pickle, zlib


def act_as_server(context, server_identity):
    """current node connection is configured as server and return zmq socket for all communication"""
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt_string(zmq.IDENTITY, server_identity)
    socket.bind("tcp://*:5555")
    return socket


def act_as_client(context, server_identity, client_identity):
    """current node connection is configured as client, attached to this round's server
    and return zmq socket for all communication"""
    socket = context.socket(zmq.DEALER)
    socket.setsockopt_string(zmq.IDENTITY, client_identity)
    server_string = "tcp://" + str(server_identity) + ":5555"
    socket.connect(server_string)
    return socket


def send_zipped_pickle(socket, obj, flags=0, protocol=pickle.HIGHEST_PROTOCOL):
    """pickle the object, compress the pickle and send it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)


def recv_zipped_pickle(socket, flags=0):
    """reverse compress and pickle operations to get object"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)


def send_updated_model(socket, obj, identity, protocol=pickle.HIGHEST_PROTOCOL):
    """used for large model files which may be sent in multiple parts"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send_multipart([identity, z])
