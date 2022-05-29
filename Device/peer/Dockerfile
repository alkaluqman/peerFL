FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
COPY peer.py ./peer.py
COPY zmq_helper.py ./zmq_helper.py
COPY training.py ./training.py
COPY inference.py ./inference.py
COPY ./saved_data_test /usr/thisdocker/testset
COPY comm_template.json ./comm_template.json

RUN pip install pyzmq
EXPOSE 5555-5565
CMD ["python", "peer.py"]