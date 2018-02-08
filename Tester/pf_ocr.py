from PIL import Image
import pytesseract
import cv2.cv2 as cv2
import os
import imutils
import pillowfight

img1 = 'cards/' + '14 HIP Prime EPO' + '.png'
img2 = 'cards/arch1.png'
img3 = 'cards/prim1.png'  # bad performance

IMAGE_PATH = img3  # '14 HIP Prime EPO.png'


img_in = Image.open(IMAGE_PATH)
x, y = img_in.size
img_in = img_in.resize((x*3, y*3), Image.ANTIALIAS)

img_in = pillowfight.unpaper_grayfilter(img_in)
# img_in = pillowfight.ace(img_in, slope=10, limit=1000, samples=100, seed=None)
# img_in = pillowfight.gaussian(img_in, sigma=2.0, nb_stddev=5)
# img_in = pillowfight.unpaper_blurfilter(img_in)
# img_in = pillowfight.unpaper_noisefilter(img_in)
img_in = pillowfight.swt(img_in, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)

img_in.show()
# todo OCR
# create image for tesseract lib
filename = "{}.tiff".format(os.getpid())
img_in.save(filename)

text = pytesseract.image_to_string(Image.open(filename), lang='eng', config='-psm 4')  # 4
os.remove(filename)  # remove created image
# print(repr(text))
print(text)
