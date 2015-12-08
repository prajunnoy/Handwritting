__author__ = 'ssitang'

import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

# read image (as opencv image)
img = cv2.imread('brown.jpg', 0)

kernel = np.ones((5, 5), np.uint8)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# dilation = cv2.dilate(img,kernel,iterations = 1)

# Canny edges detection
edges = cv2.Canny(opening, 50, 100)
# edges = cv2.Canny(img,100,200)
# edges = auto_canny(img)

# show image after applied canny edges detection
# plt.subplot(211), plt.imshow(edges,cmap = 'gray')
# plt.title('Canny Edge detection'), plt.xticks([]), plt.yticks([])
# plt.subplot(111), plt.imshow(img)

ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
count = 0
angle = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        count += 1
        # cv2.drawContours(img,cnt,-1,(0,255,0),2)
        # leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        # rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        # topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        # bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        # top.append(topmost)
        # bottom.append(bottommost)

        rect = cv2.minAreaRect(cnt)
        angle.append(rect[2])
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

        # find angle
        for ag in angle:
            if ag < -45:
                ag+=90
                print 'ver'
            elif ag == -90:
                ag+=90
                print 'same position'
            else:
                ag+=180
                print 'hor'

        # [x2, y2], no1, [x1, y1], no2 = box
        vv = cv2.boundingRect(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]
        cv2.imwrite(str(count) + '.jpg', roi)

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # x,y,w,h = cv2.boundingRect(cnt)
        # reg = (x,y,x+w,y+h)
        # coor.append(reg)

        # cv2.rectangle(img,(x, y),(x+w, y+h),(0, 255, 0), 2)
        # print x, y, w, h
        # find centroid of each character
        # M = cv2.moments(cnt)
        # centroid_x = int(M['m10']/M['m00'])
        # centroid_y = int(M['m01']/M['m00'])

        # print (leftmost)
        # print (rightmost)
        # print (topmost)
        # print (bottommost)
        # print (centroid_x)
        # print (centroid_y)

        plt.subplot(211), plt.imshow(img)
        print (rect)
        print (box)

        # roi = img[y:y+h, x:x+w]
        # cv2.imwrite(str(count) + '.jpg', roi)
        # cv2.rectangle(img, (x, y),(x+w, y+h),(0, 255, 0), 2)
        # cv2.putText(img, 'Moth Detected', (x+w+10, y+h), 0, 0.3, (0, 255, 0))

        print('done ///////////////////////////////////////////////////////////////////////// %d') %count

# rotate the pages by angle(in degree) of bottom line
# rotated = ndimage.rotate(img, angle[1])

# plt.subplot(211), plt.imshow(img)
# plt.subplot(212), plt.imshow(rotated)
# plt.title('Canny Edge detection'), plt.xticks([]), plt.yticks([])

plt.show()
