import cv2
import cnn_model
import numpy as np


def sort_contours(cnts, img_dim, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding box and use their attributes to
    # sort from top to bottom
    bounding_boxes = list()
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if (area < 100) or (area > 0.9*(img_dim[0] * img_dim[1])):
            continue
        bounding_boxes.append(cv2.boundingRect(cnt))
    cnts_boxes = zip(cnts, bounding_boxes)
    cnts_boxes = sorted(cnts_boxes, key=lambda pair: pair[1][i], reverse=reverse)
    cnts = [c for c, b in cnts_boxes]
    bounding_boxes = [b for c, b in cnts_boxes]
    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes


def preprocess_image(src_img):
    prep_imgs = list()
    loc_imgs = list()
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(gray_img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if avg_color > 127:
        print("This is a bright image")
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        print("This is a dark image")
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.blur(thresh_img, (2, 2))
    # Detect each digit in the image = bounding box
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, bounding_boxes = sort_contours(contours, thresh_img.shape)
    for box in bounding_boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        loc_imgs.append((x, y, w, h))
        cropped_img = thresh_img[y:y+h, x:x+w]
        # After crop image to 16x20, padding to 28x28
        resized_img = cv2.resize(cropped_img, (16, 20))
        padded_img = cv2.copyMakeBorder(resized_img, 4, 4, 6, 6, cv2.BORDER_CONSTANT, 0)
        # More processing
        padded_img = cv2.equalizeHist(padded_img)
        prep_imgs.append(padded_img)
    return prep_imgs, loc_imgs


def main():
    model = cnn_model.get_model()
    src_img = cv2.imread('test_images/hw_image2.png')
    prep_imgs, loc_imgs = preprocess_image(src_img)
    n = len(prep_imgs)
    result_string = ''
    label_names = cnn_model.get_label()
    for i in range(n):
        # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('output', (200, 200))
        # cv2.imshow('output', img)
        # cv2.waitKey(0)
        x_test = prep_imgs[i].reshape((1, 28, 28, 1))
        x_test = x_test.astype('float32')
        x_test = x_test / 255.0
        ans = model.predict(x_test)
        tmp = np.argmax(ans[0])
        result_string = result_string + label_names[tmp]
        # print('Predicted:', label_names[tmp])
        # print('Original ans:', ans[0])
        # Draw on image
        x, y, w, h = loc_imgs[i]
        cv2.rectangle(src_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(src_img, label_names[tmp], org=(x - 10, y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
    cv2.imshow('output', src_img)
    cv2.waitKey(0)
    print(result_string)
    cv2.destroyAllWindows()


# MAIN CODE START HERE
if __name__ == "__main__":
    main()

# TODO: Order of words
