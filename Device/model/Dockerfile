FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3
COPY . .

RUN pip install pyzmq
EXPOSE 5555-5565
CMD ["python", "model/panm.py"]
