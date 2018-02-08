import numpy as np
import imutils
import glob
import cv2
import os

img1 = '../cards/14 HIP Prime EPO.png'
img2 = '../cards/arch1.png'
img3 = '../cards/prim1.png'
img4 = '../cards/empire2.png' # vraca emblem usrani jebem mu mater
img5 = '../cards/prim2.png' # opet emblem, jebeni blueshild kurac ne radi
img6 = '../cards/prim3.png'
img7 = '../cards/visa2.png'
IMAGE_PATH = img5
TEMPLATE_DIR_PATH = '../templates' + "/*.png"

imList = [img1, img2, img3, img4, img5, img6, img7]
result_list = []
# load the image, convert it to grayscale, and initialize the
# bookkeeping variable to keep track of the matched region
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = imutils.resize(gray, height=800)
found = None

# loop over the images to find the template in
for templatePath in glob.glob(TEMPLATE_DIR_PATH):
    # get template name
    template_name = os.path.splitext(os.path.basename(templatePath))[0]
    # print(template_name)
    template = cv2.imread(templatePath)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)

    template = imutils.resize(template)
    (tH, tW) = template.shape[:2]
    # cv2.imshow("Template", template)
    # cv2.waitKey(0)
    # print('Template shape: ', template.shape[:2])
    # cv2.imshow("Template", template)
    # cv2.waitKey(0)

    # loop over the scales of the image
    for scale in np.linspace(0.1, 1, 30)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))

        r = image.shape[1] / float(resized.shape[1])

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
            found = (maxVal, maxLoc, r, template_name)


# unpack the bookkeeping varaible and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r, t) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

print('Result: ', t)
# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)





# for imagePath in imList:
#     print(imagePath)
#     # load the image, convert it to grayscale, and initialize the
#     # bookkeeping variable to keep track of the matched region
#     image = cv2.imread(imagePath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     found = None
#
#     # loop over the scales of the image
#     for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#         # resize the image according to the scale, and keep track
#         # of the ratio of the resizing
#         resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
#         r = gray.shape[1] / float(resized.shape[1])
#
#         # if the resized image is smaller than the template, then break
#         # from the loop
#         if resized.shape[0] < tH or resized.shape[1] < tW:
#             break
#         # detect edges in the resized, grayscale image and apply template
#         # matching to find the template in the image
#         edged = cv2.Canny(resized, 50, 200)
#         result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
#         (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
#
#         # check to see if the iteration should be visualized
#
#         # draw a bounding box around the detected region
#         clone = np.dstack([edged, edged, edged])
#         cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
#                       (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#         cv2.imshow("Visualize", clone)
#         cv2.waitKey(0)
#
#         # if we have found a new maximum correlation value, then ipdate
#         # the bookkeeping variable
#         if found is None or maxVal > found[0]:
#             found = (maxVal, maxLoc, r)
#
#             # unpack the bookkeeping varaible and compute the (x, y) coordinates
#             # of the bounding box based on the resized ratio
#     (_, maxLoc, r) = found
#     (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
#     (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
#
#     # draw a bounding box around the detected result and display the image
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#     cv2.imshow("Image", image)
#
# cv2.waitKey(0)

