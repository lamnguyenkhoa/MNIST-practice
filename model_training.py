import keras
import numpy as np
import os
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, Input
from keras.optimizers import SGD
from sklearn.model_selection import KFold, train_test_split
import tensorflow.python.util.deprecation as deprecation
from choose_dataset import get_dataset, DatasetEnum


def normalize_image(image_data):
    # Normalize into 0-1
    image_data = image_data.astype('float32')
    image_data = image_data / 255.0
    return image_data


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


def create_inception_model(n_label):
    ...


def image_augment():
    ...


def quick_train_and_evaluate(x_data, y_data, label_names, save):
    model = create_vgg_model(len(label_names))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1)
    y_predict = model.predict(x_test)
    if save:
        filepath = "saved_model"
        model.save(filepath)
    y_write = np.vstack([np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)])
    y_write = np.transpose(y_write)
    np.savetxt("pred_true_log.csv", y_write, delimiter=",", fmt='%d')
    return model


def kfold_train_and_evaluate(x_data, y_data, n_folds, label_names, save):
    print("=======TRAIN AND EVALUATE===========")
    histories = list()
    bestScore = 0
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    best_model = None
    for train_index, test_index in kfold.split(x_data):
        # define model
        model = create_homemade_model(len(label_names))
        x_train, y_train = x_data[train_index], y_data[train_index]
        x_test, y_test = x_data[test_index], y_data[test_index]
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print('> %.3f' % (acc * 100.0))
        histories.append(history)
        if acc > bestScore:
            best_model = model
            bestScore = acc
    if save:
        best_model.save('saved_model')
    print('==============================')
    return histories, best_model


def get_trained_model():
    model = keras.models.load_model('saved_model')
    return model


def train_model(dataset_used):
    x_data, y_data, label_names = get_dataset(dataset_used)
    print("Finished loading data")
    x_data = normalize_image(x_data)
    model = quick_train_and_evaluate(x_data, y_data, label_names, save=True)
    return model


def main():
    train_model(dataset_used=DatasetEnum.MNIST_AZ)


# MAIN CODE START HERE
if __name__ == "__main__":
    # make Tensorflow use GPU instead of CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # stop deprecation warning
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main()

# TODO: Implement image augmentation
