import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.python.util.deprecation as deprecation


def load_data():
    # Load EMNIST-balanced dataset. Merge both train and test set into one
    train_x = []
    train_y = []
    for row in open("emnist_data/emnist-balanced-train.csv"):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        train_y.append(label)
        train_x.append(image)
    for row in open("emnist_data/emnist-balanced-test.csv"):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        train_y.append(label)
        train_x.append(image)
    n = len(train_x)
    train_y = to_categorical(train_y)
    train_x = np.array(train_x).reshape((n, 28, 28, 1))
    train_y = np.array(train_y)
    return train_x, train_y


def normalize_image(image_data):
    # Normalize into 0-1
    image_data = image_data.astype('float32')
    image_data = image_data / 255.0
    return image_data


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())  # keep data in range(0,1)
    model.add(MaxPool2D((2, 2)))  # down-sampling/ reduce size of data
    model.add(Dropout(0.2))  # randomly drop 20% input to prevent over-fitting
    model.add(Flatten())  # flatten into 1d for Dense layers
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))  # speedup training and simplify model
    model.add(BatchNormalization())  # keep data in range(0,1)
    model.add(Dense(47, activation='softmax'))  # normalize output into probability distribution (47 labels)
    opti = SGD(lr=0.01, momentum=0.9, decay=0.02/10)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def image_augment():
    ...


def quick_train_and_evaluate(data_x, data_y, save=False):
    model = create_model()
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1)
    model.fit(train_x, train_y, epochs=10, batch_size=64, validation_data=(test_x, test_y), verbose=1)
    model.evaluate(test_x, test_y, verbose=1)
    if save:
        model.save("saved_model")


def kfold_train_and_evaluate(data_x, data_y, n_folds=5, save=False):
    print("=======TRAIN AND EVALUATE===========")
    histories = list()
    bestScore = 0
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    best_model = None
    for train_index, test_index in kfold.split(data_x):
        # define model
        model = create_model()
        train_x, train_y = data_x[train_index], data_y[train_index]
        test_x, test_y = data_x[test_index], data_y[test_index]
        history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=0)
        _, acc = model.evaluate(test_x, test_y, verbose=0)
        print('> %.3f' % (acc * 100.0))
        histories.append(history)
        if acc > bestScore:
            best_model = model
            bestScore = acc
    if save:
        best_model.save('saved_model')
    print('==============================')
    return histories, best_model


def get_label():
    label_names = []
    for row in open("emnist_data/emnist-balanced-mapping.txt"):
        num = int(row.split()[1])
        label_names.append(chr(num))
    return label_names


def get_model(load=True):
    # TODO: Take into account skewed images
    if load:
        model = keras.models.load_model('saved_model')
    else:
        label_names = get_label()
        train_x, train_y = load_data()
        print("Finished loading data")
        train_x = normalize_image(train_x)
        model = kfold_train_and_evaluate(train_x, train_y, n_folds=5, save=True)
    return model


def main():
    get_model(load=False)


# MAIN CODE START HERE
if __name__ == "__main__":
    # make Tensorflow use GPU instead of CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # stop deprecation warning
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main()
