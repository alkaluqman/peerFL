import pickle
import zlib
import socket
import sys

def act_as_server(server_id):
    s = socket.socket()
    s.bind(("", 5555))
    s.listen(100)
    return s

def get_ip(node_id):
    return f"10.12.{int(node_id[-1])//250}.{int(node_id[-1])%250}"

def send_zipped_pickle(s, data, protocol = pickle.HIGHEST_PROTOCOL):
    
    print("#################SANE################## : 1000000000000000")
    c, addr = s.accept()
    print("#################SANE################## : 2000000000000000")
    print(sys.getsizeof(data))
    #data = np.arange(1000000)
    p = pickle.dumps(data, protocol)
    print(sys.getsizeof(data))
    #z = zlib.compress(p)
    
    print("#################SANE################## : 3000000000000000")
    c.send(p)
    print("#################SANE################## : 4000000000000000")
    c.close()
    

def act_as_client():
    s = socket.socket()
    return s

def recv_zipped_pickle(s, from_node ,size = 1024*1024):
    server_ip = get_ip(from_node)
    s.connect((f"{server_ip}", 5555))
    msg = bytes(0)
    i= 0
    ### Recieving loop
    while True:
        print(f"recv: {i}")
        i += 1
        data = s.recv(size)
        if data:
            msg += data
        else:
            break
    #p = zlib.decompress(msg)
    return pickle.loads(msg)


  