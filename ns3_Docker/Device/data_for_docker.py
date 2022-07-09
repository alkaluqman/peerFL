import numpy as np
import random
import os
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def create_and_save_clients(num_clients=10):
    X_train, y_train, X_test, y_test = load_dataset()
    X_train, X_test = prep_pixels(X_train, X_test)
    clients = create_clients(X_train, y_train, num_clients=10, initial='client')

    basepath = os.path.join(os.getcwd(), "all_data")
    os.makedirs(basepath, 0o777)
    # process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data)
        # print(type(clients_batched[client_name]), len(clients_batched[client_name]))
        # saving the dataset to disk
        client_filename = "saved_data_" + client_name
        client_path = os.path.join(basepath, client_filename)
        tf.data.experimental.save(clients_batched[client_name], client_path)
    # print(clients_batched)

    # process and batch the test set
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    # print(type(test_batched), len(test_batched))
    client_filename = "saved_data_test"
    client_path = os.path.join(basepath, client_filename)
    tf.data.experimental.save(test_batched, client_path)

def load_clients(num_clients):
    basepath = os.path.join(os.getcwd(), "all_data")
    for i in range(1,num_clients+1):
        client_path = os.path.join(basepath, "saved_data_client_"+str(i))
        print("Loading from {} ".format(client_path))
        new_dataset = tf.data.experimental.load(client_path)
        print(len(new_dataset))

if __name__ == '__main__':
    # Create clients and save each dataset
    create_and_save_clients()


    # Load back and check client datasets
    #load_clients(6)
