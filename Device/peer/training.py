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
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import joblib
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, InputLayer
from keras.models import Sequential
from keras.models import Model


"""
ResNet-18
Reference:
[1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In
ICCV, 2015.
"""


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)

        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class SimpleMLP:

    @staticmethod
    def build(shape, classes, only_digits=True):
        #base_model_1 = VGG16(include_top=False, input_shape=shape, classes=classes)
        #base_model_1 = MobileNetV2(include_top=False, input_shape=shape, classes=classes)
        #base_model_1.trainable = False
        #base_model_1 = tf.keras.models.load_model('/usr/thisdocker/dataset/base_model')
        #base_model_1.trainable = False
        model_1 = Sequential()
        model_1.add(InputLayer(input_shape=shape))
      #  model_1.add(Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal"))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(64))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(64))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(128, down_sample=True))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(128))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(256, down_sample=True))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(256))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(512, down_sample=True))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(ResnetBlock(512))  # Adds the base model (in this case vgg19 to model_1)
      #  model_1.add(GlobalAveragePooling2D())
        model_1.add(Flatten())  # Since the output before the flatten layer is a matrix we have to use this function to get a
        # vector of the form nX1 to feed it into the fully connected layers
        # Add the Dense layers along with activation and batch normalization
        #model_1.add(Dense(1024, activation=('relu'), input_dim=512))
        #model_1.add(Dense(512, activation=('relu')))
        #model_1.add(Dense(256, activation=('relu')))
        #model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
        model_1.add(Dense(128, activation=('relu')))
        # model_1.add(Dropout(.2))
        model_1.add(Dense(classes, activation=('softmax')))  # This is the classification layer
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
    client_path = "/usr/thisdocker/dataset"
    print("[INFO] Loading from {} ".format(client_path))
    
    new_dataset = tf.data.experimental.load(client_path)
    return new_dataset


def local_training(client_num, local_model, build_flag):
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    
    return local_model

def build_model(client_num, local_model, build_flag):
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

    # Save model - moved to node
    
    return local_model