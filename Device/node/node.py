import zmq

def act_as_server():
    return None

def act_as_client():
    return None

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")