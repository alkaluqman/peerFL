import zmq
import os

context = zmq.Context()

#  Socket to talk to server
print("[CLIENT] Connecting to hello world server…")
socket = context.socket(zmq.REQ)
zmq_address = os.environ["ZMQ_ADDRESS"]
socket.connect(zmq_address)
# socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("[CLIENT] Sending request %s …" % request)
    socket.send(b"Hello")

    #  Get the reply.
    message = socket.recv()
    print("[CLIENT] Received reply %s [ %s ]" % (request, message))
