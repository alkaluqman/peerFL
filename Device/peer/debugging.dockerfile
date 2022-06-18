FROM ns_base

COPY debugging.py ./debugging.py
COPY ns_helper.py ./ns_helper.py
WORKDIR /ns/bake/source/ns-3.32/

CMD ["./waf" ,"--pyrun", "/ns/bake/source/ns-3.32/debugging.py"]