import keras
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import choose_dataset
from choose_dataset import get_label, get_dataset, DatasetEnum
from model_training import normalize_image
import matplotlib.pyplot as plt


def visualise_sample():
    x_data, y_data = choose_dataset.get_mnist_az_data()
    print(x_data.shape)
    print(y_data.shape)
    for i in range(len(x_data)):
        if np.argmax(y_data[i]) == 16:  # character G
            fig = plt.figure()
            plt.imshow(x_data[i], cmap='gray')
            plt.show()


def model_predict_to_log():
    """Load a model and use it to evaluate to save time"""
    model = keras.models.load_model('trained_models')
    x_data, y_data, label_names = get_dataset(DatasetEnum.MNIST_AZ)
    x_data = normalize_image(x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    y_predict = model.predict(x_test)
    y_write = np.vstack([np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)])
    y_write = np.transpose(y_write)
    np.savetxt("log/pred_true_log.csv", y_write, delimiter=",", fmt='%d')


def analyse_log():
    y_test = []
    y_predict = []
    for row in open("log/pred_true_log.csv"):
        row = row.split(",")
        y_test.append(int(row[0]))
        y_predict.append(int(row[1]))
    n = len(y_test)
    tmp = int(n/2)
    label_names = get_label(DatasetEnum.MNIST_AZ)
    print(classification_report(y_test, y_predict, target_names=label_names))
    cm = confusion_matrix(y_test, y_predict)
    np.savetxt("log/confusion_matrix_log.csv", cm, delimiter=",", fmt='%d')


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    visualise_sample()