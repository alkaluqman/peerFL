# moddified from https://github.com/booksbyus/zguide/blob/master/examples/Python/asyncio_ioloop/asyncsrv.py

import asyncio
import sys

import zmq
from zmq.asyncio import Context, Poller

# FRONTEND_ADDR = 'tcp://*:5570'
FRONTEND_ADDR = "tcp://*:5559"
BACKEND_ADDR = "tcp://*:5560"
# FRONTEND_ADDR = "inproc://frontend"
# BACKEND_ADDR = "inproc://backend"


class Client:
    """A client that generates requests."""

    def __init__(self, context, id):
        self.context = context
        self.id = id

    async def run_client(self):
        socket = self.context.socket(zmq.REQ)
        identity = "client-%d" % self.id
        socket.connect(FRONTEND_ADDR)
        print("Client %s started" % (identity))
        reqs = 0
        while True:
            reqs = reqs + 1
            msg = f"request # {self.id}.{reqs}"
            msg = msg.encode("utf-8")
            await socket.send(msg)
            print(f"Client {self.id} sent request: {reqs}")
            msg = await socket.recv()
            print(f"Client {identity} received: {msg}")
            await asyncio.sleep(1)


class Server:
    """A server to set up and initialize clients and request handlers"""

    def __init__(self, loop, context):
        self.loop = loop
        self.context = context

    def run_server(self):
        tasks = []
        frontend = self.context.socket(zmq.ROUTER)
        frontend.bind(FRONTEND_ADDR)
        backend = self.context.socket(zmq.DEALER)
        backend.bind(BACKEND_ADDR)
        task = run_proxy(frontend, backend)
        tasks.append(task)

        # Start up the workers.
        for idx in range(5):
            worker = Worker(self.context, idx)
            task = worker.run_worker()
            tasks.append(task)

        # Start up the clients.
        tasks += [Client(self.context, idx).run_client() for idx in range(3)]
        return tasks


class Worker:
    """A request handler"""

    def __init__(self, context, idx):
        self.context = context
        self.idx = idx

    async def run_worker(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect(BACKEND_ADDR)
        print(f"Worker {self.idx} started")
        while True:
            ident, part2, msg = await worker.recv_multipart()
            print(f"Worker {self.idx} received {msg} from {ident}")
            await asyncio.sleep(0.5)
            await worker.send_multipart([ident, part2, msg])
        worker.close()


async def run_proxy(socket_from, socket_to):
    poller = Poller()
    poller.register(socket_from, zmq.POLLIN)
    poller.register(socket_to, zmq.POLLIN)
    while True:
        events = await poller.poll()
        events = dict(events)
        if socket_from in events:
            msg = await socket_from.recv_multipart()
            await socket_to.send_multipart(msg)
        elif socket_to in events:
            msg = await socket_to.recv_multipart()
            await socket_from.send_multipart(msg)


def run(loop):
    context = Context()
    server = Server(loop, context)
    tasks = server.run_server()
    loop.run_until_complete(asyncio.wait(tasks))


def main():
    """main function"""
    print("(main) starting")
    args = sys.argv[1:]
    if len(args) != 0:
        sys.exit(__doc__)
    try:
        loop = asyncio.get_event_loop()
        run(loop)
    except KeyboardInterrupt:
        print("\nFinished (interrupted)")


if __name__ == "__main__":
    main()
