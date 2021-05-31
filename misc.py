# Contain unused code


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
