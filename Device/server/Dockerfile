FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
COPY router.py ./router.py
COPY inference.py ./inference.py
COPY ./saved_data_test /usr/thisdocker/testset
RUN pip install pyzmq
EXPOSE 5555
CMD ["python", "router.py"]