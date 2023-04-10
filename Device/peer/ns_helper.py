import pickle
import zlib
import socket
import sys
import time

def act_as_server(server_id):
    s = socket.socket()
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind(("", 5555))
    s.listen(100)
    return s

def get_ip(node_id):
    return f"10.12.{int(node_id[-1])//250}.{int(node_id[-1])%250}"

def send_zipped_pickle(s, data, protocol = pickle.HIGHEST_PROTOCOL):
    
    print("### ATTEMPTING CONNECTION")
    c, addr = s.accept()
    print(f"### CONNECTED TO {addr}")
    #print(sys.getsizeof(data))
    #data = np.arange(1000000)
    p = pickle.dumps(data, protocol)
    #print(sys.getsizeof(p))
    #z = zlib.compress(p)
    
    print(f"### ATTEMPTING DATA SEND of {sys.getsizeof(p)}")
    c.send(p)
    print("### DATA SENT SUCCESSFULLY")
    c.close()
    #s.close()
    

def act_as_client():
    s = socket.socket()
    return s

def recv_zipped_pickle(s, from_node ,size = 1024):
    server_ip = get_ip(from_node)
    s.connect((f"{server_ip}", 5555))
    msg = []
    i= 0
    #bytes_recv = 0
    ### Recieving loop
    done = False
    tot = 0
    #time.sleep(60)
    print("### RECEIVING")
    while True:
        #print(f"recv: {i}")
        size = 1024*1024
        i += 1
        
        a =time.time()
        data = s.recv(size)
        b = time.time()
        if data:
            #tmp = len(data)
            #size = min(size, tmp)
            msg.append(data)
            tot += float(b-a)
        else:
            break
    print(tot)
    print(float(tot/i))
    msg =b''.join(msg)
    print(sys.getsizeof(msg))
    print("### RECEIVED")
    
    print(i)
    #s.close()
    #p = zlib.decompress(msg)
    return pickle.loads(msg)


