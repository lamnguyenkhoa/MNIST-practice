import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical


class DatasetEnum:
    """ Enum class for choosing which dataset"""
    MNIST_AZ = 0  # Usually good. Contain lots of uppercase
    EMNIST_BALANCE = 1  # Not very useful
    MNIST_EMNIST_LETTER = 2  # Fast while still good. Contain cursive letter
    EMNIST_BYMERGE = 3  # Load very slow


def get_data_description(val):
    desc = "Unknown dataset"
    if val == 0:
        desc = "Trained using the combined MNIST dataset with A-Z dataset"
    if val == 1:
        desc = "Trained using the balanced dataset"
    if val == 2:
        desc = "Trained using combined MNIST dataset with EMNIST Letter dataset"
    if val == 3:
        desc = "Trained using EMNIST by merge dataset."
    return desc


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


def get_mnist_az_data():
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
    x_data, y_data = read_data_from_csv("training_data/emnist-balanced-train.csv")
    x, y = read_data_from_csv("training_data/emnist-balanced-test.csv")
    x_data.extend(x)
    y_data.extend(y)
    n = len(x_data)
    x_data = np.array(x_data).reshape((n, 28, 28, 1))
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return x_data, y_data


def get_mnist_emnist_letter_data():
    # Load EMNIST-letter only dataset. Merge both train and test set into one
    x_data, y_data = read_data_from_csv("training_data/emnist-letters-train.csv")
    x, y = read_data_from_csv("training_data/emnist-letters-test.csv")
    x_data.extend(x)
    y_data.extend(y)
    y_data = [y-1 for y in y_data]  # The labels start at 1 so need to minus by 1
    x_data = np.array(x_data)
    x_data = x_data.swapaxes(1, 2)  # Rotate data by 90 degree clockwise
    # Avoid same labels as mnist digits
    for i in range(len(y_data)):
        y_data[i] += 10
    # Load MNIST data
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    mnist_x_data = np.vstack([x_train, x_test])
    mnist_y_data = np.hstack([y_train, y_test])
    # Merge both of them
    x_data = np.vstack([x_data, mnist_x_data])
    y_data = np.hstack([y_data, mnist_y_data])
    n = len(x_data)
    x_data = x_data.reshape((n, 28, 28, 1))
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return x_data, y_data


def get_emnist_by_merge_data():
    # Load EMNIST-bymerge dataset. Merge both train and test set into one
    x_data, y_data = read_data_from_csv("training_data/emnist-bymerge-train.csv")
    x, y = read_data_from_csv("training_data/emnist-bymerge-test.csv")
    x_data.extend(x)
    y_data.extend(y)
    n = len(x_data)
    x_data = np.array(x_data).reshape((n, 28, 28, 1))
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return x_data, y_data


def get_label(val):
    label_names = None
    if (val == DatasetEnum.EMNIST_BALANCE) or (val == DatasetEnum.EMNIST_BYMERGE):
        label_names = []
        for row in open("training_data/emnist-balanced-mapping.txt"):
            num = int(row.split()[1])
            label_names.append(chr(num))
    if (val == DatasetEnum.MNIST_AZ) or (val == DatasetEnum.MNIST_EMNIST_LETTER):
        label_names = "0123456789"
        label_names += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        label_names = [str(c) for c in label_names]
    return label_names


def get_dataset(val):
    """
    :param val: Enum for dataset
    :return: x_data, y_data, label_names
    """
    x_data, y_data = None, None
    label_names = get_label(val)
    if val == DatasetEnum.MNIST_AZ:
        x_data, y_data = get_mnist_az_data()
    if val == DatasetEnum.EMNIST_BALANCE:
        x_data, y_data = get_emnist_balanced_data()
    if val == DatasetEnum.MNIST_EMNIST_LETTER:
        x_data, y_data = get_mnist_emnist_letter_data()
    if val == DatasetEnum.EMNIST_BYMERGE:
        x_data, y_data = get_emnist_by_merge_data()
    return x_data, y_data, label_names
