import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical


class DatasetEnum:
    """ Enum class for choosing which dataset"""
    MNIST_AZ = 1
    EMNIST_BALANCE = 2


def read_data_from_csv(filepath):
    x = []
    y = []
    for row in open(filepath):
        row = row.split(',')
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        y.append(label)
        x.append(image)
    return x, y


def get_mnist_and_az_data():
    # Load Kaggle A-Z letter data
    az_x_data, az_y_data = read_data_from_csv("training_data/A_Z Handwritten Data.csv")
    # Avoid same labels as mnist digits
    for i in range(len(az_y_data)):
        az_y_data[i] += 10

    # Load MNIST data
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    mnist_x_data = np.vstack([x_train, x_test])
    mnist_y_data = np.hstack([y_train, y_test])

    # Merge both of them
    x_data = np.vstack([az_x_data, mnist_x_data])
    y_data = np.hstack([az_y_data, mnist_y_data])
    n = len(x_data)
    x_data = np.array(x_data).reshape((n, 28, 28, 1))
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return x_data, y_data


def get_emnist_balanced_data():
    # Load EMNIST-balanced dataset. Merge both train and test set into one
    x_data = []
    y_data = []
    x, y = read_data_from_csv("training_data/emnist-balanced-train.csv")
    x_data.extend(x)
    y_data.extend(y)
    x, y = read_data_from_csv("training_data/emnist-balanced-test.csv")
    x_data.extend(x)
    y_data.extend(y)
    n = len(x_data)
    x_data = np.array(x_data).reshape((n, 28, 28, 1))
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return x_data, y_data


def get_label(val):
    label_names = None
    if val == DatasetEnum.EMNIST_BALANCE:
        label_names = []
        for row in open("training_data/emnist-balanced-mapping.txt"):
            num = int(row.split()[1])
            label_names.append(chr(num))
    if val == DatasetEnum.MNIST_AZ:
        label_names = "0123456789"
        label_names += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        label_names = [str(c) for c in label_names]
    return label_names


def get_dataset(val):
    x_data, y_data = None, None
    label_names = get_label(val)
    if val == DatasetEnum.MNIST_AZ:
        x_data, y_data = get_mnist_and_az_data()
    if val == DatasetEnum.EMNIST_BALANCE:
        x_data, y_data = get_emnist_balanced_data()
    return x_data, y_data, label_names
