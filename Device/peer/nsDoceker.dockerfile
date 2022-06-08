FROM base_image
COPY peer.py ./peer.py
COPY training.py ./training.py
COPY inference.py ./inference.py
COPY ./saved_data_test /usr/thisdocker/testset


CMD ["./waf" , "--pyrun", "usr/thisdocker/peer.py", "--ns=True"]