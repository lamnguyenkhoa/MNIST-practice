import keras
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
import tensorflow.python.util.deprecation as deprecation
from choose_dataset import get_dataset, DatasetEnum, DatasetDescription
from choose_model import ModelEnum


def normalize_image(image_data):
    # Normalize into 0-1
    image_data = image_data.astype('float32')
    image_data = image_data / 255.0
    return image_data


def image_augment():
    ...


def quick_train_and_evaluate(x_data, y_data, model, label_names, save, model_name):
    filepath = "trained_models/" + model_name
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1)
    y_predict = model.predict(x_test)
    if save:
        model.save(filepath)
    # Write to log files
    y_write = np.vstack([np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)])
    y_write = np.transpose(y_write)
    np.savetxt("pred_true_log.csv", y_write, delimiter=",", fmt='%d')
    np.savetxt(filepath + "/labels.txt", label_names, newline=' ', fmt='%s')
    with open(filepath + "/info.txt", "w") as f:
        f.write(DatasetDescription.MNIST_AZ)
    return model


def get_trained_model(model_name):
    filepath = "trained_models/" + model_name
    model = keras.models.load_model(filepath)
    with open (filepath + "/labels.txt") as f:
        label_names = f.readline(). split(' ')
    return model, label_names


def train_model(dataset_used, model_used):
    """
    After training, save the model in /trained_models folder
    """
    x_data, y_data, label_names = get_dataset(dataset_used)
    print("Finished loading data")
    x_data = normalize_image(x_data)
    trained_model = quick_train_and_evaluate(x_data, y_data, model_used, label_names, True, "homemade_model")
    return trained_model


def main():
    train_model(dataset_used=DatasetEnum.MNIST_EMNIST_LETTER,
                model_used=ModelEnum.HOMEMADE_MODEL)


# MAIN CODE START HERE
if __name__ == "__main__":
    # make Tensorflow use GPU instead of CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # stop deprecation warning
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main()

# TODO: Implement image augmentation
