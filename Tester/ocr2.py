import tempfile
import numpy as np
from PIL import Image
import pytesseract
import cv2.cv2 as cv2
import cv2 as cvekla
import os
import matplotlib.pyplot as plt

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


# Pokusaj1

def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def tesser(gray):
    filename = "{}.tiff".format(os.getpid())
    cv2.imwrite(filename, gray)

    text = pytesseract.image_to_string(Image.open(filename), lang='eng', config='-psm 4')  # 4
    os.remove(filename)  # remove created image
    return text


if __name__ == '__main__':
    img1 = 'cards/' + '14 HIP Prime EPO' + '.png'
    img2 = 'cards/arch1.png'
    img3 = 'cards/prim1.png'  # bad performance

    # img = process_image_for_ocr(img3)
    # cv2.imshow("Splacina", img)
    # cv2.waitKey(0)
    # t = tesser(img)
    # print(t)

    # img = cv2.imread(img1)
    # img = cv2.GaussianBlur(img, (5, 5), 1)  # smooth image
    # mask = cv2.inRange(img, (40, 180, 200), (70, 220, 240))  # filter out yellow color range, low and high range
    # gray = 255 - mask
    text = pytesseract.image_to_string(Image.open('slikicica.png'), lang='eng', config='-psm 4')
    print(text)

