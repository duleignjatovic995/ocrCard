from PIL import Image
import pytesseract
import cv2.cv2 as cv2
import os
import imutils
import pillowfight as pf

img1 = 'cards/' + '14 HIP Prime EPO' + '.png'
img2 = 'cards/arch1.png'
img3 = 'cards/prim1.png'  # bad performance

IMAGE_PATH = img3  # '14 HIP Prime EPO.png'
PREPROCESS = True

# Reading image
image = cv2.imread(IMAGE_PATH)

# Resizing image
image = imutils.resize(image, height=800, inter=cv2.INTER_CUBIC)

# Converting image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting image to gray scale

# todo PREPROCESSING IMAGE
if PREPROCESS is True:
    # todo Resize image
    gray = imutils.resize(gray, height=600, inter=cv2.INTER_CUBIC)

    # todo Blurring
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # todo Binarize image
    # gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.CALIB_CB_ADAPTIVE_THRESH,
                                 thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)


    # todo Tophat (ideja: sa invert slikom izvuci sto bolje preho hat metoda)
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # create kernel
    # gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)  # problematican buljas

    # todo Edge detection
    # gray = cv2.Canny(gray, 50, 200)

    # todo Blurring
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # todo Blackhat
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    # sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    #
    # gray = cv2.bitwise_not(gray)

    # todo Denoising funkcija
    # gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

# todo OCR
# create image for tesseract lib
filename = "{}.tiff".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename), lang='eng', config='-psm 4')  # 4
os.remove(filename)  # remove created image
# print(repr(text))
print(text)

# todo NLP: NER


# todo PRINTING


# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)

cv2.waitKey(0)
