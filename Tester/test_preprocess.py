from PIL import Image
import pytesseract
import cv2.cv2 as cv2
import util
from util import *
import os
import imutils
import numpy as np
import pillowfight
from imutils import skeletonize

img1 = 'cards/14 HIP Prime EPO.png'
img2 = 'cards/arch1.png'
img3 = 'cards/prim1.png'
img4 = 'cards/empire2.png' # meh
img5 = 'cards/prim2.png'
img6 = 'cards/prim3.png'
img7 = 'cards/visa2.png'

IMAGE_PATH = img1
PREPROCESS = True


def prep1(gray):
    gray = adaptive_histogram(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return gray


def prep2(gray):
    gray = adaptive_histogram(gray)
    # gray = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.threshold(gray, 140, 255, cv2.THRESH_OTSU)[1]
    return gray


def prep3(gray):  # suplja slova, solidan rezultat
    for i in range(6):
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.CALIB_CB_ADAPTIVE_THRESH,
                                     thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)

        # if i % 3 == 0:
        #     gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                  thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
        #     gray = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO)[1]
    # gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # gray = cv2.erode(gray, kernel, iterations=1)
    return gray


def prep4(gray):  # suplje malo
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
    gray = swt(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    # gray = cv2.erode(gray, kernel, iterations=1)
    return gray


def prep5(gray):  # djubre
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=1)

    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return gray


def prep6(gray):  # blue cmss :(
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    return gray


def prep7(gray):
    gray = adaptive_histogram(gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.CALIB_CB_ADAPTIVE_THRESH,
    #                              thresholdType=cv2.THRESH_BINARY, blockSize=3, C=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    # gray = cv2.erode(gray, kernel, iterations=2)
    return gray


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def prep8(file_name):
    # img = cv2.imread(file_name, 0)
    img = file_name
    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def sharpen3(gray):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray


def sharpen4(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return gray


def final_preprocessor(gray):
    """
    sharpen1->ada_hist->otsu-> dilate(kernel ell 2,1)
    """

    gray = sharpen1(gray)
    # gray = sharpen2(gray)
    # gray = sharpen3(gray)
    # gray = sharpen4(gray)
    # cv2.imshow("1", gray1)
    # cv2.imshow("2", gray2)
    # cv2.imshow("3", gray3)
    # cv2.imshow("4", gray4)

    gray = adaptive_histogram(gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))  # (10, 2) grebe horizontalno
    gray = cv2.dilate(gray, kernel, iterations=2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1))
    # gray = cv2.erode(gray, kernel, iterations=2)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    return gray




image, gray = get_image(IMAGE_PATH)
gray = final_preprocessor(gray)


# height, width = gray.shape[:2]
# print(height, width)
# gray = gray[0:(height//3), :] #width//2

# image = cv2.imread(IMAGE_PATH)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = adaptive_histogram(gray)
# gray = imutils.resize(gray, height=800, inter=cv2.INTER_CUBIC)
# kurac = image_smoothening(gray)
# gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
# gray = cv2.bitwise_or(kurac, gray)



# todo EXPERIMENTAL
# gray = prep1(gray)
# gray = prep2(gray)
# gray = prep3(gray)
# gray = prep4(gray)
# gray = prep5(gray)  # wasted
# gray = prep6(gray)
# gray = prep7(gray)
# gray = prep8(gray)
# gray = final_preprocessor(gray)

# print('')

# gray = adaptive_histogram(gray)
# gray = blur(gray)
# gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.CALIB_CB_ADAPTIVE_THRESH,
#                              thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
# # gray = swt(gray)
# gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

# gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # create kernel
# gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)  # problematican buljas
# todo Edge detection
# gray = cv2.Canny(gray, 50, 200)
# todo Blackhat
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

# sqKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
# gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, sqKernel)
# gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=2)
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# image = cv2.bilateralFilter(image, 5, 150, 50)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              thresholdType=cv2.THRESH_BINARY, blockSize=3, C=2)
# sqKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, sqKernel)
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

# gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              thresholdType=cv2.THRESH_BINARY, blockSize=3, C=2)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)


# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# gray = cv2.dilate(gray, kernel, iterations=1)
# gray = cv2.erode(gray, kernel, iterations=1)



# todo SWT
# gray = swt(gray)

# todo Blurring
# gray = blur(gray)

# todo Histogram equilization
# gray = adaptive_histogram(gray)


# todo Binarize image
# gray = cv2.threshold(gray, 240, 255, cv2.THRESH_TRUNC)[1]
# gray = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO)[1]
# gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.CALIB_CB_ADAPTIVE_THRESH,
#                              thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
# gray = swt(gray)

# TODO ITERACIJA bin + thresh -> daje rezultat/ bez clahe za sad
# for i in range(6):
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#     gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.CALIB_CB_ADAPTIVE_THRESH,
#                                  thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
#
#     if i % 3 == 0:
#         gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
#                                      thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
#         gray = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO)[1]


# todo noise removal/ losi rezultati za sad
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
# sure background area
# gray = cv2.dilate(gray, kernel, iterations=1)


# todo Tophat (ideja: sa invert slikom izvuci sto bolje preho hat metoda)
# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # create kernel
# gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)  # problematican buljas
# todo Edge detection
# gray = cv2.Canny(gray, 50, 200)
# todo Blackhat
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
# gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
#
# gray = cv2.bitwise_not(gray)

# todo Denoising funkcija
# gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

print('Preprocessing done.')

# todo OCR
raw_text = my_ocr(gray)
print(raw_text)

# todo IMSHOW
# cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
