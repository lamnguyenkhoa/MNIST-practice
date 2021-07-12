import numpy as np
import os
import tensorflow.python.util.deprecation as deprecation
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from choose_dataset import get_dataset, DatasetEnum, get_data_description
from choose_model import get_model, ModelEnum
import matplotlib.pyplot as plt


def normalize_image(image_data):
    # Normalize into 0-1
    image_data = image_data.astype('float32')
    image_data = image_data / 255.0
    return image_data


def plot_model_result(histories, filepath):
    # summarize history for accuracy
    plt.plot(histories.history['accuracy'])
    plt.plot(histories.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(filepath + "/model_accuracy.png")
    plt.show()  # show() should come after after savefig()
    # summarize history for loss
    plt.plot(histories.history['loss'])
    plt.plot(histories.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(filepath + "/model_loss.png")
    plt.show()


def quick_train_and_evaluate(x_data, y_data, model, with_aug=False):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    # Create image augmentation
    img_aug = ImageDataGenerator(zoom_range=0.2,
                                 shear_range=0.15,
                                 width_shift_range=[-2, 2],
                                 height_shift_range=[-2, 2])
    if with_aug:
        n = len(x_train)
        histories = model.fit(x=img_aug.flow(x_train, y_train, batch_size=64), epochs=15,
                              validation_data=(x_test, y_test), verbose=2, steps_per_epoch=n/64)
    else:
        histories = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2)
    y_predict = model.predict(x_test)
    # Write to log files
    y_write = np.vstack([np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)])
    y_write = np.transpose(y_write)
    np.savetxt("log/pred_true_log.csv", y_write, delimiter=",", fmt='%d')
    return model, histories


def get_trained_model(model_name):
    """
    Load a trained model from /trained_models.
    Not to confuse with get_model(), which is load the model skeleton in choose_model()
    """
    filepath = "trained_models/" + model_name
    model = load_model(filepath)
    with open(filepath + "/labels.txt") as f:
        label_names = f.readline(). split(' ')
    return model, label_names


def train_model(dataset_used, model_used, model_name, with_aug):
    """
    After training, save the model in /trained_models folder
    """
    x_data, y_data, label_names = get_dataset(dataset_used)
    model = get_model(model_used, len(label_names))
    print("Finished loading data")
    x_data = normalize_image(x_data)
    trained_model, histories = quick_train_and_evaluate(x_data, y_data, model, with_aug)

    # Save and log
    filepath = "trained_models/" + model_name
    trained_model.save(filepath)
    plot_model(model, to_file=filepath+'/model_diagram.png', show_shapes=True, show_layer_names=True)
    np.savetxt(filepath + "/labels.txt", label_names, newline=' ', fmt='%s')
    with open(filepath + "/info.txt", "w") as f:
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S\n"))  # Log datetime
        f.write(get_data_description(dataset_used)+"\n")
        if with_aug:
            f.write("Used image augmentation\n")
    plot_model_result(histories, filepath)
    return trained_model


def main():
    train_model(dataset_used=DatasetEnum.EMNIST_BYMERGE,
                model_used=ModelEnum.HOMEMADE_MODEL,
                model_name="homemade_model3",
                with_aug=False)


# MAIN CODE START HERE
if __name__ == "__main__":
    # Stop notification
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # stop deprecation warning
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    main()

# TODO: Train something w/ EMNIST_BYMERGE
# TODO: train_test_split() set test_size = 0.2 result in error in small dataset???
# TODO: current models has way too high variance
