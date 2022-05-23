FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
COPY dealer.py ./dealer.py
COPY training.py ./training.py
RUN pip install pyzmq
EXPOSE 5555
CMD ["python", "dealer.py"]