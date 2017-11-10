from PIL import Image
import pytesseract
import cv2.cv2 as cv2
import os
import imutils
import numpy as np
import pillowfight
import io
import base64


def get_image(path):
    # Reading image
    image = cv2.imread(path)
    image, gray = format_image(image)
    return image, gray


def format_image(image):
    # Resizing image
    image = imutils.resize(image, height=800, inter=cv2.INTER_CUBIC)
    # Converting image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def swt(img):
    img = Image.fromarray(img)
    img = pillowfight.swt(img, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)
    img = np.array(img)
    return img


def blur(img, type='g'):
    if type == 'g':
        img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        img = cv2.medianBlur(img, 3)
    return img


def adaptive_histogram(img, adaptive=True):
    if adaptive is True:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    else:
        img = cv2.equalizeHist(img)  # regular hist

    return img


def sharpen1(gray): # best for now
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel_sharpen)
    return gray


def sharpen2(gray):
    kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    gray = cv2.filter2D(gray, -1, kernel_sharpen)
    return gray


def image_preprocessor(gray):
    gray = sharpen1(gray)
    gray = adaptive_histogram(gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))
    gray = cv2.dilate(gray, kernel, iterations=2)
    return gray

#  -oem 0 textord_heavy_nr 1
# enable_new_segsearch 1
# load_system_dawg F
# load_freq_dawg F
# 'textord_min_linesize 3.25' \
# 'textord_heavy_nr 1' \



def my_ocr(gray):
    CONFIG = '--psm 1 ' \
             '--oem 2 ' \
             'tessedit_char_whitelist 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:' \
             'textord_space_size_is_variable 1'
    # create image for tesseract lib
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename), lang='eng',
                                       config='--psm 1',
                                       )
    os.remove(filename)  # remove created image
    # print(repr(text))
    return text


