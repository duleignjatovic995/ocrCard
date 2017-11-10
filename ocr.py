"""
The image processing and OCR, along with template matching for
"""
# import util
import cv2
import numpy as np
import imutils
import pytesseract
import os
import glob
from PIL import Image

TEMPLATE_DIR_PATH = 'templates' + "/*.png"


def get_text(image_path):
    image, gray = get_image(image_path)
    processed_image = image_preprocessor(gray)
    text = my_ocr(processed_image)
    return text


def get_insurer(image_path):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = imutils.resize(gray, height=800)
    found = None

    # loop over the images to find the template in
    for templatePath in glob.glob(TEMPLATE_DIR_PATH):
        # get template name
        template_name = os.path.splitext(os.path.basename(templatePath))[0]
        template = cv2.imread(templatePath)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        template = imutils.resize(template)
        (tH, tW) = template.shape[:2]

        # loop over the scales of the image
        for scale in np.linspace(0.1, 1, 30)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))

            # r = image.shape[1] / float(resized.shape[1])

            # get upper part of image
            height, width = resized.shape[:2]
            resized = resized[0:(height // 3), :]

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                # print('Template larger than image!', template_name)
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, template_name)
    (_, maxLoc, template) = found
    return template


def image_preprocessor(gray):
    gray = sharpen1(gray)
    gray = adaptive_histogram(gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))
    gray = cv2.dilate(gray, kernel, iterations=2)
    return gray


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
