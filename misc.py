from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np
from choose_dataset import get_label
from choose_dataset import DatasetEnum
from choose_dataset import get_dataset
from model_training import normalize_image
import keras
import os
import matplotlib.pyplot as plt


def model_predict_to_log():
    """Load a model and use it to evaluate to save time"""
    model = keras.models.load_model('saved_model')
    x_data, y_data, label_names = get_dataset(DatasetEnum.MNIST_AZ)
    x_data = normalize_image(x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    y_predict = model.predict(x_test)
    y_write = np.vstack([np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)])
    y_write = np.transpose(y_write)
    np.savetxt("pred_true_log.csv", y_write, delimiter=",", fmt='%d')


def analyse_log():
    y_test = []
    y_predict = []
    for row in open("pred_true_log.csv"):
        row = row.split(",")
        y_test.append(int(row[0]))
        y_predict.append(int(row[1]))
    n = len(y_test)
    tmp = int(n/2)
    label_names = get_label(DatasetEnum.MNIST_AZ)
    print(classification_report(y_test, y_predict, target_names=label_names))
    cm = confusion_matrix(y_test, y_predict)
    np.savetxt("confusion_matrix_log.csv", cm, delimiter=",", fmt='%d')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_predict_to_log()
analyse_log()
