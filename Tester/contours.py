# import the necessary packages
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
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))


# image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# smooth the image using a 3x3 Gaussian, then apply the blackhat
# morphological operator to find dark regions on a light background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("Blackhat", blackhat)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
cv2.imshow("gradX Sobel", gradX)

gradX = np.absolute(gradX)
cv2.imshow("gradX Absolute", gradX)

(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
cv2.imshow("gradX jebada", gradX)

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2.imshow("gradX Close", gradX)

thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh otsu", thresh)

# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow("CLOSE", thresh) # ne sad
thresh = cv2.erode(thresh, None, iterations=4)
cv2.imshow("ERODE", thresh) # ne sad

cv2.imshow("Kurac", image)
# cv2.imshow(":D", thresh)
cv2.waitKey(0)
