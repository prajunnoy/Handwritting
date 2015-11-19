import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from matplotlib import pyplot as plt

img = cv2.imread('new.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray,(5,5),0)
# edges = cv2.Canny(img, 50, 100)
# contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img,contours,-1,(255,255,255),-1)

# thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret3,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)

cv2.imwrite('final.jpg', thresh)

plt.subplot(221),plt.imshow(img)
# plt.title('Original Image')
plt.subplot(222),plt.imshow(imgray,'gray')
# plt.title('Extracted extra region Image')
plt.subplot(223),plt.imshow(thresh, 'gray')
# plt.title('Cropped Image')
plt.show()
