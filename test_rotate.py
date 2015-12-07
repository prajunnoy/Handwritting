from scipy import ndimage
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

####################################################################
# read image (as opencv image)
img = cv2.imread('final.jpg',0)

kernel = np.ones((5,5),np.uint8)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# dilation = cv2.dilate(img,kernel,iterations = 1)

# Canny edges detection
edges = cv2.Canny(opening,50,100)
# edges = cv2.Canny(img,100,200)
# edges = auto_canny(img)

# show image after applied canny edges detection
# plt.subplot(211),plt.imshow(edges,cmap = 'gray')
# plt.title('Canny Edge detection'), plt.xticks([]), plt.yticks([])
# plt.subplot(111),plt.imshow(img)

ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
count = 0
coor = []
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

        # rect = cv2.minAreaRect(cnt)
        # box = cv2.cv.BoxPoints(rect)
        # box = np.int0(box)

        # cv2.drawContours(img,[box],0,(0,255,0),2)

        x,y,w,h = cv2.boundingRect(cnt)
        reg = (x,y,x+w,y+h)
        coor.append(reg)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # print x, y, w, h
        # find centroid of each character
        # M = cv2.moments(cnt)
        # centroid_x = int(M['m10']/M['m00'])
        # centroid_y = int(M['m01']/M['m00'])

        # print (leftmost)
        # print (rightmost)
        # print (topmost)
        # print (bottommost)
        # print(centroid_x)
        # print(centroid_y)

        # print(rect)
        # print(box)

        # roi = img[y:y+h, x:x+w]
        # cv2.imwrite(str(count) + '.jpg', roi)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(img,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))

        print('done ///////////////////////////////////////////////////////////////////////// %d') %count

height, width = img.shape
print img.shape
# sort coordinate of x1 ascendingly
coor = sorted(coor, key=lambda tup: tup[0])
# most-valued of y-axis
# top = sorted(top, key=lambda tup: tup[0])
# bottom = sorted(bottom, key=lambda tup: tup[0])
print coor

# Find the bottom line from left to rightmost
# checking condition that y1 and y2 is between
# y1 and y2 of previous coordinate
idx = 0
idx_arr = []
# idx_b = 0
y11 = height/2
y21 = height
# checking condition for bottom line
for x1, y1, x2, y2 in coor:

	print idx, coor[idx], y11, y21
	print x1, y1, x2, y2

	if x1==1 or x2==1:
		# coor.pop(idx)
		idx_arr.append(idx)

		# if y1 not btw y11andy22 or y2 not btw y11and y22
	elif y11 <= y1 <= y21 or y11 <= y2 <= y21:
		y11 =  y1
		y21 =  y2

	else:
		# coor.pop(idx)
		idx_arr.append(idx)

	idx+=1

print idx_arr

# pop unwanted coordinate
a =0
for i in idx_arr:
	i-=a
	coor.pop(i)
	a+=1

print coor

# Draw rectangle of bottom line (not work) mayb sort har t la tua
print len(coor)
# x1, yy1, xx1, y1 = coor[0]
# xx2, y2, x2, yy2 = coor[len(coor)-1]
# cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

# for sort (x1 y1 x2 y2)
# sort x1 from ascending order (need x1 and y2)
coor = sorted(coor, key=lambda tup: tup[0])
X1, Y1, X2, Y2 = coor[0]
print coor[0]
# sort x2 from descending order (need x2 and y2)
coor = sorted(coor, key=lambda tup: tup[2], reverse=True)
XX1, YY1, XX2, YY2 = coor[0]
print coor[0]

# find angle from H and P
H = XX2 - X1
if YY2 > Y2:
	P = YY2 - Y2
elif Y2 > YY2:
	P = Y2 - YY2
print XX2, X1
print YY2, Y2
print H, P
print P/float(H)
angle = math.asin(P/float(H))
# convert angle from radians to degrees
angle = math.degrees(angle)
print angle

# rotate the pages by angle(in degree) of bottom line
rotated = ndimage.rotate(img, 360-angle, mode='nearest')

# Crop the bottom line image
# cut = img[289:474, 38:2832]
# cv2.imwrite('cut.jpg', cut)
# cv2.imwrite(str(count) + '.jpg', roi)
cv2.imwrite('rotate.jpg', rotated)
plt.subplot(211),plt.imshow(img)
plt.subplot(212),plt.imshow(rotated)

# cv2.imshow('img',img)
# cv2.waitKey(0)
plt.show()
