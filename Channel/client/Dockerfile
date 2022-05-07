FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
COPY client.py ./client.py
RUN pip install pyzmq
EXPOSE 5555
CMD ["python", "client.py"]