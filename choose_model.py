import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, Input
from keras.optimizers import SGD


class ModelEnum:
    HOMEMADE_MODEL = 0
    VGG_MODEL = 1


def create_homemade_model(n_label):
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())  # keep data in range(0,1)
    model.add(MaxPool2D((2, 2)))  # down-sampling/ reduce size of data
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))  # randomly drop 20% input to prevent over-fitting
    model.add(Flatten())  # flatten into 1d for Dense layers
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))  # speedup training and simplify model
    model.add(Dense(n_label, activation='softmax'))  # normalize output into probability distribution
    opti = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def vgg_block(layer_in, n_filters, n_conv):
    """
    Function for creating a vgg block
    Copied from https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
    """
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    # add max pooling layer
    layer_in = MaxPool2D((2, 2), strides=(2, 2))(layer_in)
    return layer_in


def create_vgg_model(n_label):
    input_layer = Input(shape=(28, 28, 1))
    vgg1_layer = vgg_block(input_layer, 32, 2)
    vgg2_layer = vgg_block(vgg1_layer, 64, 2)
    flatten_layer = Flatten()(vgg2_layer)
    output_layer = Dense(n_label, activation='softmax')(flatten_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    opti = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model(val, n_label):
    if val == ModelEnum.HOMEMADE_MODEL:
        return create_homemade_model(n_label)
    if val == ModelEnum.VGG_MODEL:
        return create_vgg_model(n_label)
