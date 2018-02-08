import cv2
import numpy as np
import imutils

img = cv2.imread('../cards/prim1.png')
# print(img.shape[:2])

img = imutils.resize(img, height=800, inter=cv2.INTER_CUBIC)
# Converting image to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# cv2.imshow('Original', img)


# generating	the	kernels

kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0
# applying	different	kernels	to	the	input	image
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
cv2.imshow('Sharpening', output_1)  # ovaj
# cv2.imshow('Excessive	Sharpening', output_2)
cv2.imshow('Edge	Enhancement', output_3)  # ovaj
cv2.waitKey(0)
