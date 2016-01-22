# For skeletonize the image
import scipy.ndimage.morphology as m
from skimage import morphology
import numpy as np
import cv2

def skeletonize(img):
    h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]])
    m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]])
    h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]])
    m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))
    img = img.copy()
    while True:
        last = img
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            img = np.logical_and(img, np.logical_not(hm))
        if np.all(img == last):
            break
    return img

img = cv2.imread("sentence10.jpg", 0)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
img_bin = thresh
# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# dilated = cv2.dilate(img_bin, element, iterations=3)

# This is extra code for deleting skeleton small branch
# By using scikit image process
# dilated = cv2.dilate(img_bin, element)
# dilated = cv2.dilate(dilated, element)

skel = morphology.skeletonize(img_bin > 0)
# skel = skel.astype(float)
#############################################

# skel = skeletonize(dilated)
skel = skel.astype(np.uint8)*255
final = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
print final

cv2.imwrite('skel.jpg', final)
cv2.imshow('skel imge', skel)
cv2.waitKey()