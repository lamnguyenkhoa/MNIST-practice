import keras
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow.python.util.deprecation as deprecation
from choose_dataset import get_dataset
from choose_dataset import get_label
from choose_dataset import DatasetEnum


def normalize_image(image_data):
    # Normalize into 0-1
    image_data = image_data.astype('float32')
    image_data = image_data / 255.0
    return image_data


def create_model(n_label):
    model = Sequential()
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


def image_augment():
    ...


def quick_train_and_evaluate(x_data, y_data, label_names, save):
    model = create_model(len(label_names))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1)
    y_predict = model.predict(x_test)
    if save:
        model.save("saved_model")
    np.savetxt("pred_true_log.txt", np.argmax(y_predict, axis=1), delimiter=',', fmt='%d')
    np.savetxt("pred_true_log.txt", np.argmax(y_test, axis=1), delimiter=',', fmt='%d')
    return model


def kfold_train_and_evaluate(x_data, y_data, n_folds, label_names, save):
    print("=======TRAIN AND EVALUATE===========")
    histories = list()
    bestScore = 0
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    best_model = None
    for train_index, test_index in kfold.split(x_data):
        # define model
        model = create_model(len(label_names))
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


def get_model(load, dataset_used):
    if load:
        model = keras.models.load_model('saved_model')
    else:
        x_data, y_data, label_names = get_dataset(dataset_used)
        print("Finished loading data")
        x_data = normalize_image(x_data)
        model = quick_train_and_evaluate(x_data, y_data, label_names, save=True)
    return model


def main():
    get_model(load=False, dataset_used=DatasetEnum.MNIST_AZ)


# MAIN CODE START HERE
if __name__ == "__main__":
    # make Tensorflow use GPU instead of CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # stop deprecation warning
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main()

# TODO: Implement image augmentation
