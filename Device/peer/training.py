import numpy as np
import os
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
import joblib

class SimpleMLP:

    @staticmethod
    def build(shape, classes, only_digits=True):
        base_model_1 = VGG16(include_top=False, input_shape=shape, classes=classes)
        model_1 = Sequential()
        model_1.add(base_model_1)  # Adds the base model (in this case vgg19 to model_1)
        model_1.add(
            Flatten())  # Since the output before the flatten layer is a matrix we have to use this function to get a
        # vector of the form nX1 to feed it into the fully connected layers
        # Add the Dense layers along with activation and batch normalization
        model_1.add(Dense(1024, activation=('relu'), input_dim=512))
        model_1.add(Dense(512, activation=('relu')))
        model_1.add(Dense(256, activation=('relu')))
        # model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
        model_1.add(Dense(128, activation=('relu')))
        # model_1.add(Dropout(.2))
        model_1.add(Dense(10, activation=('softmax')))  # This is the classification layer
        return model_1


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clients
    global_count = sum(
        [tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(1 * weight[i])
    return weight


def scale_model_weights2(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(1 * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def load_client_dataset():
    # basepath = os.path.join(os.getcwd(), "all_data")
    # client_path = os.path.join(basepath, "saved_data_client_"+str(client_num))
    client_path = "/usr/thisdocker/dataset"
    print("[INFO] Loading from {} ".format(client_path))
    new_dataset = tf.data.experimental.load(client_path)
    return new_dataset


# if __name__ == '__main__':
def local_training(client_num, local_model, build_flag):
    # Load client dataset from volume mounted folder
    # client_num = 1
    log_prefix = "[" + str(client_num).upper() + "] "
    local_dataset = load_client_dataset()
    x = local_dataset.element_spec[0].shape[1]
    y = local_dataset.element_spec[0].shape[2]
    z = local_dataset.element_spec[0].shape[3]
    input_shape = (x, y, z)
    num_classes = local_dataset.element_spec[1].shape[1]

    if build_flag:
        # Create model
        lr = 0.01
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        optimizer = SGD(lr=lr, decay=lr, momentum=0.9)

        print("%sBuilding model ..." % log_prefix)
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(shape=input_shape, classes=num_classes)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        local_model.build(input_shape=(None, x, y, z))

    print("%sTraining model ..." % log_prefix)
    # Training
    local_model.fit(local_dataset, epochs=1, verbose=1)
    print("%sDone" % log_prefix)

    # Save model - moved to node
    return local_model
