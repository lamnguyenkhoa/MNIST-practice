import cv2
import numpy as np
import model_training


def display_image_cv2(img, window_name="output"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 400)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def ratio_resize(img):
    """
    Resized an image while keep its ratio
    """
    (old_h, old_w) = img.shape[:2]
    # First, make the image shape become square
    if old_w > old_h:
        diff = old_w - old_h
        if diff % 2 == 0:
            img = cv2.copyMakeBorder(img, diff // 2, diff // 2, 0, 0, cv2.BORDER_CONSTANT, 0)
        else:
            img = cv2.copyMakeBorder(img, diff // 2, diff - (diff // 2), 0, 0, cv2.BORDER_CONSTANT, 0)

    if old_h > old_w:
        diff = old_h - old_w
        if diff % 2 == 0:
            img = cv2.copyMakeBorder(img, 0, 0, diff // 2, diff // 2, cv2.BORDER_CONSTANT, 0)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, diff // 2, diff - (diff // 2), cv2.BORDER_CONSTANT, 0)
    # Then resize it normally
    resized_img = cv2.resize(img, (20, 20), cv2.INTER_AREA)  # Chose 16,20 because the horizontal sides are more empty
    return resized_img


def sort_contours(cnts, img_dim, method):
    """ Copied from imutils package"""
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
    bounding_boxes = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        box = cv2.boundingRect(cnt)
        if (area < 0.0005 * (img_dim[0] * img_dim[1])) \
                or (area < 100) \
                or (area > 0.9 * (img_dim[0] * img_dim[1])):
            continue  # Ignore abnormal contours
        bounding_boxes.append(box)
    cnts_boxes = zip(cnts, bounding_boxes)
    cnts_boxes = sorted(cnts_boxes, key=lambda pair: pair[1][i], reverse=reverse)
    cnts = [c for c, _ in cnts_boxes]
    bounding_boxes = [b for _, b in cnts_boxes]
    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes


def separate_vertical_lines(src_img, visual=False):
    """ This function try to find lines of words and sort them"""
    line_imgs = []
    loc_lines = []
    # Turn image into grayscale: black background with white color font
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(gray_img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if avg_color > 127:
        print("This is a bright image")
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        print("This is a dark image")
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    blur_img = cv2.blur(thresh_img, (32, 2))  # Horizontal blur to detect word lines
    if visual:
        display_image_cv2(blur_img, "blur_img in separate_vertical_lines")
    # Detect each digit in the image = bounding box
    contours, _ = cv2.findContours(blur_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, bounding_boxes = sort_contours(contours, blur_img.shape, "top-to-bottom")
    for box in bounding_boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        loc_lines.append((x, y, w, h))
        cropped_img = thresh_img[y:y + h, x:x + w]
        padded_img = cv2.copyMakeBorder(cropped_img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, 0)
        if visual:
            display_image_cv2(padded_img, "word lines")
        line_imgs.append(padded_img)
    return line_imgs, loc_lines


def preprocess_image(src_img, visual=False):
    """ This function assume the src_img contain 1 word """
    prep_imgs = []
    loc_imgs = []
    blur_img = cv2.blur(src_img, (1, 4))  # Blur to detect 2-parts letters such as i and j
    if visual:
        display_image_cv2(blur_img, "blur_img in preprocess_image")
    # Detect each digit in the image = bounding box
    contours, _ = cv2.findContours(blur_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, bounding_boxes = sort_contours(contours, blur_img.shape, "left-to-right")
    for box in bounding_boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        loc_imgs.append((x, y, w, h))
        cv2.rectangle(src_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cropped_img = src_img[y:y + h, x:x + w]
        # After resize image to 16x20, padding to 28x28
        resized_img = ratio_resize(cropped_img)
        padded_img = cv2.copyMakeBorder(resized_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, 0)
        # More processing
        padded_img = cv2.blur(padded_img, (2, 2))
        padded_img = cv2.equalizeHist(padded_img)
        prep_imgs.append(padded_img)
    return prep_imgs, loc_imgs


def main():
    model, label_names = model_training.get_trained_model("homemade_model4")
    src_img = cv2.imread('test_images/hw_image4.jpg')
    line_imgs, loc_lines = separate_vertical_lines(src_img)
    id_counter = 0
    result_string = ""
    for i in range(len(line_imgs)):
        prep_imgs, loc_imgs = preprocess_image(line_imgs[i])
        for j in range(len(prep_imgs)):
            # display_image_cv2(prep_imgs[j])
            x_test = prep_imgs[j].reshape((1, 28, 28, 1))
            x_test = x_test.astype('float32')
            x_test = x_test / 255.0
            pred = model.predict(x_test)[0]
            tmp = np.argmax(pred)
            prob = pred[tmp]
            # Print probability info for each word
            print("[Id {} Prob. ] {} - {:.2f}%".format(id_counter, label_names[tmp], prob * 100))

            result_string = result_string + label_names[tmp]
            # Draw on image
            x = loc_lines[i][0] + loc_imgs[j][0] - 25  # minus the padding bonus of separate_vertical_lines()
            y = loc_lines[i][1] + loc_imgs[j][1] - 25
            w, h = loc_imgs[j][2:4]
            cv2.rectangle(src_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(src_img, label_names[tmp], org=(x - 5, y - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.putText(src_img, "id" + str(id_counter), org=(x + 10, y - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)
            id_counter += 1
        result_string += " "  # Space to separate word lines
    print(result_string)
    display_image_cv2(src_img)


# MAIN CODE START HERE
if __name__ == "__main__":
    main()

# TODO: Order of words horizontally
# TODO: Classify in 1, l, i, j
# TODO: Classify 9 vs g, 6 vs G
# TODO: Assume scr_image is suitable size for better contours detection
