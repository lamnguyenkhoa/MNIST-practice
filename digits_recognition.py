import cv2
import mnist_cnn
import numpy as np


def deskew(img):
    return ...


def get_blob_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def preprocess_image(filename, display=False):
    prep_imgs = list()
    new_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(gray_img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if avg_color > 127:
        print("This is a bright image")
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        print("This is a dark image")
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    # Detect each digit in the image = bounding box
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # If contour is too small or too large then skip
        area = cv2.contourArea(cnt)
        if (area < 100) or (area > 0.9*(thresh_img.shape[0] * thresh_img.shape[1])):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_img = thresh_img[y:y+h, x:x+w]
        # After crop image to 18x18, padding 5px each side to 28x28
        resized_img = cv2.resize(cropped_img, (18, 18))
        padded_img = cv2.copyMakeBorder(resized_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, 0)
        prep_imgs.append(padded_img)
    if display:
        cv2.imshow('output', new_img)
        cv2.waitKey(0)
    return prep_imgs


def main():
    model = mnist_cnn.get_model()
    prep_imgs = preprocess_image('hw_digits2.png', True)
    result_number = ''
    for img in prep_imgs:
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', img)
        cv2.waitKey(0)
        x_test = img.reshape((1, 28, 28, 1))
        x_test = x_test.astype('float32')
        x_test = x_test / 255.0
        ans = model.predict(x_test)
        result_number = str(np.argmax(ans[0])) + result_number
        print('Predicted:', np.argmax(ans[0]))
        # print('Original ans:', ans[0])
    print(result_number)
    cv2.destroyAllWindows()

# Main code start here
main()
