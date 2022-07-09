FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
# Tensorflow 2.8, min 2.7 required for dataset save, load functions to work
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY training.py ./training.py
COPY inference.py ./inference.py
COPY all_data/saved_data_client_1 ./all_data/saved_data_client_1
COPY all_data/saved_data_test ./all_data/saved_data_test

RUN python training.py