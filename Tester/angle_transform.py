from imutils import perspective
import numpy as np
import cv2
import imutils
from skimage.filters import threshold_adaptive

image = cv2.imread("probna.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")

cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = perspective.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# warped = cv2.adaptiveThreshold(warped, 251, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,5)
warped = threshold_adaptive(warped, 251, offset=10)
warped = warped.astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")

cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)

# # load the notecard code image, clone it, and initialize the 4 points
# # that correspond to the 4 corners of the notecard
# notecard = cv2.imread("probna.jpg")
# clone = notecard.copy()
# pts = np.array([(73, 239), (356, 117), (475, 265), (187, 443)])
#
# # loop over the points and draw them on the cloned image
# for (x, y) in pts:
#     cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
#
# # apply the four point tranform to obtain a "birds eye view" of
# # the notecard
# warped = perspective.four_point_transform(notecard, pts)
#
# # show the original and warped images
# cv2.imshow("Original", clone)
# cv2.imshow("Warped", warped)
# cv2.waitKey(0)