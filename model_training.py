import keras
import numpy as np
import os
import tensorflow.python.util.deprecation as deprecation
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split
from choose_dataset import get_dataset, DatasetEnum, get_data_description
from choose_model import get_model, ModelEnum
import matplotlib.pyplot as plt


def normalize_image(image_data):
    # Normalize into 0-1
    image_data = image_data.astype('float32')
    image_data = image_data / 255.0
    return image_data


def plot_model_result(histories, filepath, save_to_disk=True):
    # summarize history for accuracy
    plt.plot(histories.history['accuracy'])
    plt.plot(histories.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(filepath + "/model_accuracy.png")
    # summarize history for loss
    plt.plot(histories.history['loss'])
    plt.plot(histories.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filepath + "/model_loss.png")
    plt.show()


def image_augment():
    ...


def quick_train_and_evaluate(x_data, y_data, model):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    histories = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)
    y_predict = model.predict(x_test)
    # Write to log files
    y_write = np.vstack([np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)])
    y_write = np.transpose(y_write)
    np.savetxt("log/pred_true_log.csv", y_write, delimiter=",", fmt='%d')
    return model, histories


def get_trained_model(model_name):
    """
    Load a trained model from /trained_models
    Not to confuse with get_model(), which is load the model skeleton in choose_model()
    :param model_name:
    :return:
    """
    filepath = "trained_models/" + model_name
    model = keras.models.load_model(filepath)
    with open(filepath + "/labels.txt") as f:
        label_names = f.readline(). split(' ')
    return model, label_names


def train_model(dataset_used, model_used, model_name):
    """
    After training, save the model in /trained_models folder
    """
    x_data, y_data, label_names = get_dataset(dataset_used)
    model = get_model(model_used, len(label_names))
    print("Finished loading data")
    x_data = normalize_image(x_data)
    trained_model, histories = quick_train_and_evaluate(x_data, y_data, model)

    # Save and log
    filepath = "trained_models/" + model_name
    trained_model.save(filepath)
    np.savetxt(filepath + "/labels.txt", label_names, newline=' ', fmt='%s')
    with open(filepath + "/info.txt", "w") as f:
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S\n"))  # Log datetime
        f.write(get_data_description(dataset_used)+"\n")
    plot_model_result(histories, filepath)
    return trained_model


def main():
    train_model(dataset_used=DatasetEnum.MNIST_EMNIST_LETTER,
                model_used=ModelEnum.HOMEMADE_MODEL,
                model_name="homemade_model2")


# MAIN CODE START HERE
if __name__ == "__main__":
    # make Tensorflow use GPU instead of CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # stop deprecation warning
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main()

# TODO: Implement image augmentation
