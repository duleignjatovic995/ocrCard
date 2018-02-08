from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

img1 = 'cards/' + '14 HIP Prime EPO' + '.png'
img2 = 'cards/arch1.png'
img3 = 'cards/prim1.png'  # bad performance
img4 = 'cards/empire2.png'
IMAGE_PATH = img1

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# smooth the image using a 3x3 Gaussian, then apply the blackhat
# morphological operator to find dark regions on a light background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("Blackhat", blackhat)

# thresh = cv2.adaptiveThreshold(src=blackhat, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=3, C=2)
# cv2.imshow("trheshadap", thresh)

tr = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh otsu", tr)

# img = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, rectKernel)
# cv2.imshow("open", img)
# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# image = cv2.erode(image, sqKernel, iterations=1)
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, sqKernel)
image = cv2.Laplacian(gray, cv2.CV_16S, 3)
cv2.imshow("krajnja", image)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
# gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# cv2.imshow("gradX Sobel", gradX)
#
# gradX = np.absolute(gradX).astype("uint8")
# cv2.imshow("gradX Absolute", gradX)
#
# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
# cv2.imshow("gradX jebada", gradX)


# thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("Thresh otsu", thresh)

# thresh = cv2.dilate(gradX, None, iterations=1)
# cv2.imshow("ERODE", thresh)

# thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("Thresh otsu", thresh)
# thresh = cv2.adaptiveThreshold(src=thresh, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=3, C=2)

# cv2.imshow("Thresh otsu", thresh)


# cnts, _, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# loop over the contours
# for c in cnts:
#     cv2.drawContours(mask, [c], -1, 0, -1)
#     print(c)
# print(len(cnts))






cv2.waitKey(0)
