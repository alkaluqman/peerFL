import time
import zmq
import os

context = zmq.Context()
socket = context.socket(zmq.REP)
zmq_address = os.environ["ZMQ_ADDRESS"]
socket.bind(zmq_address)
print("[SERVER] Server waiting for connection")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("[SERVER] Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")
