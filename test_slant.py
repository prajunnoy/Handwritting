# import cv2
# import numpy as np
#
# img = cv2.imread('z.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# # cv2.imshow('edges', edges)
# minLineLength = 10
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength,maxLineGap)
# print lines
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imwrite('houghlines5.jpg',img)
# cv2.imshow('img',img)
# cv2.waitKey()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

img = cv2.imread('a.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,10)
print lines
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    # b is slant angle
    print a,b
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    print x1, x2,y1, y2

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',img)


#rotation angle in degree
# rotated = ndimage.rotate(img, -40.514246764)

# plt.imshow(rotated)
plt.imshow(img)
plt.show()