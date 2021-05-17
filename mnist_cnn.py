import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.python.util.deprecation as deprecation


def load_data():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    print("train_x size:", train_x.shape, "| train_y size:", train_y.shape)
    print("test_x size:", test_x.shape, "| test_y size:", test_y.shape)
    im_width = train_x.shape[1]
    im_height = train_x.shape[2]
    # Reshape train_x, test_x so it have 1 channel color
    train_x = train_x.reshape((train_x.shape[0], im_width, im_height, 1))
    test_x = test_x.reshape((test_x.shape[0], im_width, im_height, 1))
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y


def normalize_image(train_x, test_x):
    # Normalize into 0-1
    train_norm = train_x.astype('float32')
    test_norm = test_x.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())  # keep data in range(0,1)
    model.add(MaxPool2D((2, 2)))  # down-sampling/ reduce size of data
    model.add(Dropout(0.2))  # randomly drop 20% input to prevent over-fitting
    model.add(Flatten())  # flatten into 1d for Dense layers
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))  # help speedup training
    model.add(BatchNormalization())  # keep data in range(0,1)
    model.add(Dense(10, activation='softmax'))  # normalize output into probability distribution
    opti = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(data_x, data_y, n_folds=5):
    print("=======TRAIN AND EVALUATE===========")
    histories = list()
    bestScore = 0
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    print("Training scores:")
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
    best_model.save('savedModel')
    print('==============================')
    return best_model, histories


def actual_test(model, test_x, test_y):
    _, acc = model.evaluate(test_x, test_y)
    return acc


def get_model(load=True):
    if load:
        model = keras.models.load_model('savedModel')
    else:
        train_x, train_y, test_x, test_y = load_data()
        train_x, test_x = normalize_image(train_x, test_x)
        model = train_and_evaluate(train_x, train_y)
    return model


def layers_visualize():
    print('=================Visualize==================')
    train_x, train_y, test_x, test_y = load_data()
    # Draw the original image
    plt.imshow(train_x[0], cmap=plt.get_cmap('gray'))
    plt.show()
    # Layer 1
    layer1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))
    draw_data = np.asarray([train_x[0]])
    draw_data = draw_data.astype('float32')
    draw_data = draw_data / 255.0
    draw_data = layer1(draw_data)
    print("Layer1 dim:", draw_data.shape)
    for i in range(draw_data.shape[3]):
        plt.subplot(4, 8, i + 1)
        plt.imshow(draw_data[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.show()
    # Layer 2
    layer2 = MaxPool2D((2, 2))
    draw_data = layer2(draw_data)
    print("Layer2 dim:", draw_data.shape)
    for i in range(draw_data.shape[3]):
        plt.subplot(4, 8, i + 1)
        plt.imshow(draw_data[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.show()
    # Layer 3
    layer3 = Flatten()
    draw_data = layer3(draw_data)
    print("Layer3 dim:", draw_data.shape)


def main():
    #layers_visualize()
    train_x, train_y, test_x, test_y = load_data()
    train_x, test_x = normalize_image(train_x, test_x)
    model = get_model()

    # Actual test the test set
    #acc = actual_test(model, test_x, test_y)
    #print("Test score:", acc)

    # NOTE: Make sure the image has black/dark background and white/light number
    # NOTE: 0 is black, 255 is white
    # NOTE: Since the training is handwritten image, work best with handwritten image
    # TODO: Take into account skewed images
    # TODO: Scale image to 18x18 and padding extra 5 pixel around the image -> better


# MAIN CODE START HERE
# make Tensorflow use GPU instead of CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# stop deprecation warning
deprecation._PRINT_DEPRECATION_WARNINGS = False
