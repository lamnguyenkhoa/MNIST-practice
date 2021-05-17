import cv2
import mnist_cnn
import numpy as np


def deskew(img):
    return ...


def preprocess_image(filename, display=False):
    new_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # x,y,w,h = cv2.boundingRect(contours[0])
    # img_boundrect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cropped_image = img[y:y+h, x:x+w]
    # cv2.imshow('output2', imgray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    dim = (28, 28)
    resized_img = cv2.resize(gray_img, dim)
    if display:
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return new_img


def main():
    model = mnist_cnn.get_model()
    prep_image = preprocess_image('number2.png')
    x_test = prep_image.reshape((1, 28, 28, 1))
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
    ans = model.predict(x_test)
    print('Predicted:', np.argmax(ans[0]))
    print('Original ans:', ans[0])


# Main code start here
main()
