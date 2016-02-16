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
import scipy.stats as stats
import scipy
# from scipy import ndimage
import math
import os

# img_path = os.path.join('c:\\', 'Users','ssitang','PycharmProjects','HandwritingProj','final samples','tang data','tang sentence20')
img_path = os.path.join('c:\\', 'Users','ssitang','PycharmProjects','HandwritingProj','final samples','tang data','8')
print img_path

values = []

for item in os.listdir(img_path):
    if '.jpg' in item:  # this could be more correctly done with os.path.splitext
        img = cv2.imread(os.path.join(img_path, '13.jpg'))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 10)
    print lines
    # print len(lines)
    if lines is None:
        break
    else:
        for rho, theta in lines[0]:
            # use a value to calculate further
            a = np.cos(theta)
            b = np.sin(theta)
            values.append(a)
            # b is slant angle
            # print 'a:',a, 'b:', b
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # print x1, x2, y1, y2

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # break

        cv2.imwrite('houghlines3.jpg', img)

values.sort()
print values
#
# meann = np.mean(values)
# print 'mean: ', meann
#
# variance = np.var(values)
# # sigma = math.sqrt(variance)
#
# sd = np.std(values)
# print 'standard-deviation: ', sd
#
# lowerbound = meann-sd
# upperbound = meann+sd
# # print lowerbound, upperbound
#
# shadedval = []
# for pnt in values:
#     if upperbound > pnt > lowerbound:
#         shadedval.append(pnt)
# # print shadedval
#
# shadedval_avg = np.mean(shadedval)
# print 'AVERAGE: ', shadedval_avg

# dot at mean point
# plt.plot(meann,0,'ro',markersize=8)
#
# # dash line for one and two standard deviation
# std_to_show = scipy.linspace(-3,3,7)
# std_x_values = meann + std_to_show*sd
# std_normal_curve_values = stats.norm.pdf(std_x_values,meann,sd)
#     # In order to plot these std lines all at once,
#     # we will make arrays with the first row as the starting points
#     # and the second row as the end points
# std_lines_x_array = scipy.vstack((std_x_values,std_x_values))
# std_lines_y_array = scipy.vstack((scipy.zeros((1,7)),std_normal_curve_values))
# handle_of_std_lines = plt.plot(std_lines_x_array,std_lines_y_array,'r--')
#
# x = np.linspace(min(values)-2, max(values)+2, 100)  #stepsize = 100
# fit = stats.norm.pdf(x, meann, sd)  #this is a fitting indeed
#
# # plt.plot(values, fit, '-o')
# # plt.hist(values, len(values), normed=True)      #use this to draw histogram of your data
# plt.plot(x, fit)
# # plt.show()