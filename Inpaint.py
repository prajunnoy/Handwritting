import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

##################  Create mask image for inpaint function  #############################
# img = Image.open('crop.jpg')
img = cv2.imread('crop.jpg')

# downscale the original image
# scale, im = downscale_image(img)

# convert PIL image to opencv image
# cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# just add median blur
img_gray = cv2.medianBlur(img_gray, 7)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)

# plt.subplot(221), plt.imshow(img_gray)
# plt.show()
# edges = cv2.Canny(img_gray, 100, 200)

# gaussian less noise than mean
thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
# mean give better result than gaussian (disconnected character) but more noises than gaussian
# thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,2)

# ret,thresh = cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY_INV)
# ostu not work
# ret3, thresh_ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(thresh,contours,-1,(255,255,255),5)

mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(mask,kernel,iterations = 2)

cv2.imwrite('mask.jpg', dilation)


##########################  Inpaint Function  ##########################################
mask = cv2.imread('mask.jpg', 0)

dst_NS = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

cv2.imwrite('inpaint.jpg', dst_NS)
# cv2.imwrite('in.jpg', dst_TELEA)

# plt.subplot(221),plt.imshow(img_gray)
# # plt.title('Original Image')
# plt.subplot(222),plt.imshow(thresh,'gray')
# # plt.title('Extracted extra region Image')
# plt.subplot(223),plt.imshow(dst_NS)
# # plt.title('Cropped Image')
# plt.show()
