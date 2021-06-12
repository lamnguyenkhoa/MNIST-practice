import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.python.util.deprecation as deprecation


def load_data():
    # Load EMNIST-balanced dataset. Merge both train and test set into one
    train_x = []
    y_train = []
    for row in open("emnist_data/emnist-balanced-train.csv"):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        y_train.append(label)
        train_x.append(image)
    for row in open("emnist_data/emnist-balanced-test.csv"):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        y_train.append(label)
        train_x.append(image)
    n = len(train_x)
    y_train = to_categorical(y_train)
    train_x = np.array(train_x).reshape((n, 28, 28, 1))
    y_train = np.array(y_train)
    return train_x, y_train


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
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))  # randomly drop 20% input to prevent over-fitting
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())  # flatten into 1d for Dense layers
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))  # speedup training and simplify model
    model.add(Dense(47, activation='softmax'))  # normalize output into probability distribution (47 labels)
    opti = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def image_augment():
    ...


def quick_train_and_evaluate(data_x, data_y, label_names, save=False):
    model = create_model()
    train_x, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1)
    model.fit(train_x, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1)
    y_predict = model.predict(x_test)
    if save:
        model.save("saved_model")
    f = open("log.txt", "w")
    f.write(y_predict.argmax(axis=1))
    f.write(y_test.argmax(axis=1))
    f.close()
    # classification_report(y_test.argmax(axis=1), y_predict.argmax(axis=1), labels=label_names)
    return model


def kfold_train_and_evaluate(data_x, data_y, n_folds=5, save=False):
    print("=======TRAIN AND EVALUATE===========")
    histories = list()
    bestScore = 0
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    best_model = None
    for train_index, test_index in kfold.split(data_x):
        # define model
        model = create_model()
        x_train, y_train = data_x[train_index], data_y[train_index]
        x_test, y_test = data_x[test_index], data_y[test_index]
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
        x_train, y_train = load_data()
        print("Finished loading data")
        x_train = normalize_image(x_train)
        model = quick_train_and_evaluate(x_train, y_train, label_names, save=True)
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
