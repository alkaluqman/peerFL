import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

tf.config.set_visible_devices([], 'GPU')
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


def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print("[INFO] Accuracy = {:.3%}".format(acc))
    print("[INFO] Loss = {}".format(loss))
    return acc, loss


def load_client_dataset():
    # basepath = os.path.join(os.getcwd(), "all_data")
    # client_path = os.path.join(basepath, "saved_data_"+str(client_num))
    client_path = "/usr/thisdocker/testset"
    print("[INFO] Loading from {} ".format(client_path))
    new_dataset = tf.data.experimental.load(client_path)
    return new_dataset

def eval_on_test_set(averaged_model):
    # Load client test set
    local_dataset = load_client_dataset()
    x = local_dataset.element_spec[0].shape[1]
    y = local_dataset.element_spec[0].shape[2]
    z = local_dataset.element_spec[0].shape[3]
    input_shape = (x, y, z)
    num_classes = local_dataset.element_spec[1].shape[1]

    #Load trained model
    # client_num = 1
    # model_filename = "client_" + str(client_num) + ".pkl"
    # local_model = joblib.load(model_filename)

    #test the SGD global model and print out metrics
    for(X_test, Y_test) in local_dataset:
        SGD_acc, SGD_loss = test_model(X_test, Y_test, averaged_model, 1)

def FedAvg(model_dict):
    #Global model creation
    smlp_global = SimpleMLP()
    comms_round = 2 #10
    lr = 0.01
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr, decay=lr / comms_round, momentum=0.9)
    global_model = smlp_global.build((32,32,3), 10)
    global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    global_model.build(input_shape=(None, 32, 32, 3))

    average_weights = global_model.get_weights()
    average_weights = np.array(average_weights)

    average_weights.fill(0)

    for client, local_model in model_dict.items():
        weights = local_model.get_weights()
        weights = np.array(weights)
        average_weights = np.add(average_weights, weights)

    average_weights = average_weights / len(model_dict)
    average_weights = average_weights.tolist()
    global_model.set_weights(average_weights)

    #evaluation accuracy
    eval_on_test_set(global_model)

    return global_model
