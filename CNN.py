import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
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
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    opti = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(data_x, data_y, n_folds=5):
    bestScore = 0
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    print("Training scores:")
    # enumerate splits
    for train_index, test_index in kfold.split(data_x):
        # define model
        model = create_model()
        # select rows for train and test
        train_x, train_y, test_x, test_y = data_x[train_index], data_y[train_index], data_x[test_index], data_y[
            test_index]
        # fit model
        model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y), verbose=0)
        # evaluate model
        _, acc = model.evaluate(test_x, test_y, verbose=0)
        print('> %.3f' % (acc * 100.0))
        if acc > bestScore:
            best_model = model
            bestScore = acc
        # stores scores
    best_model.save('savedModel')
    return best_model


def actual_test(model, test_x, test_y):
    _, acc = model.evaluate(test_x, test_y)
    return acc


def test_new_image(filename, model):
    # Read an image as grayscale
    print("Loaded new image.")
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    srcImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    dim = (28, 28)
    resized_img = cv2.resize(srcImg, dim)
    cv2.imshow('output', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # prepare data
    resized_img = resized_img.reshape((1, 28, 28, 1))
    resized_img = resized_img.astype('float32')
    resized_img = resized_img / 255.0
    ans = model.predict(resized_img)
    print('Predicted:', np.argmax(ans[0]))
    print('Original ans:', ans[0])


def step_by_step_visualize():
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


def driver(load=False, continuous=False):
    if not load:
        # Training the model
        train_x, train_y, test_x, test_y = load_data()
        train_x, test_x = normalize_image(train_x, test_x)
        model = train_and_evaluate(train_x, train_y)
    else:
        model = keras.models.load_model('savedModel')

    # Actual test the test set
    # actual_test(model, test_x, test_y)

    # Test with custom image
    if not continuous:
        test_new_image('number9.png', model)
    else:
        while True:
            print("=====================================")
            print("Type in image location (ex: path/to/location) : ")
            filename = input()
            if filename == "":
                break
            else:
                test_new_image(filename, model)

    # FIXME: Make sure the image has black/dark background and white/light number
    # FIXME: Since the training is handwritten image, work best with handwritten image
    # TODO: Take into account skewed images


# MAIN CODE START HERE
# make Tensorflow use GPU instead of CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

# driver code control the program
driver(True, True)
