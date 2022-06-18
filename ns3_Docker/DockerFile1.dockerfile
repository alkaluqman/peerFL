FROM ns_base

COPY peer.py ./peer.py
COPY ns_helper.py ./ns_helper.py

CMD ["./waf --pyrun", "/ns/bake/peer.py"]