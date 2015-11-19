import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('crop.jpg')

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_gray = cv2.GaussianBlur(img_gray,(5,5),0)

# edges = cv2.Canny(img_gray, 100, 200)
thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
# ret,thresh = cv2.threshold(edges,127,255,cv2.THRESH_BINARY_INV)
# ostu not work
# ret3,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

dst_NS = cv2.inpaint(img,thresh,3,cv2.INPAINT_NS)

cv2.imwrite('inpaint.jpg', dst_NS)

plt.subplot(221),plt.imshow(img)
# plt.title('Original Image')
plt.subplot(222),plt.imshow(thresh,'gray')
# plt.title('Extracted extra region Image')
plt.subplot(223),plt.imshow(dst_NS)
# plt.title('Cropped Image')
plt.show()
