import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from matplotlib import pyplot as plt

imgray = cv2.imread('new.jpg', 0)
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgray = cv2.GaussianBlur(imgray,(5,5),0)
# edges = cv2.Canny(img, 50, 100)
# contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img,contours,-1,(255,255,255),-1)

# the best result out of all
ret,thresh = cv2.threshold(imgray,20,255,cv2.THRESH_BINARY_INV)
# not work (result only edges of characters)
# thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# not work some characters lost (disconnected character and some parts of characters are lighter than the others)
# ret3, thresh = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

final = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
cv2.imwrite('final.jpg', final)

# plt.subplot(221),plt.imshow(img)
# # plt.title('Original Image')
# plt.subplot(222),plt.imshow(imgray,'gray')
# # plt.title('Extracted extra region Image')
plt.subplot(223),plt.imshow(final)
# plt.title('Cropped Image')
plt.show()
