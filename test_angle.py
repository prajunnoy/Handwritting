import Image, ImageDraw
import numpy as np
import collections
import itertools
from pylab import *
from scipy import *
from skimage import io
from skimage.morphology import skeletonize
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

# Load the picture with PIL, process if needed
orig_img = Image.open('skel1.jpg')
print orig_img
w, h = orig_img.size
pic = asarray(orig_img)

# Vertical Projection
# Average the pixel values along vertical axis
pic_avg_v = pic.mean(axis=2)
projection_v = pic_avg_v.sum(axis=0)
print projection_v
# Compute the variance along vertical axis
variance_v = pic_avg_v.var(axis=0)
# print variance_v

# Horizontal Projection
# Average the pixel values along horizontal axis
# pic_avg_h = pic.mean(axis=1)
# # print pic_avg_h
# projection_h = pic_avg_h.sum(axis=1)
# print projection_h
# # Compute the variance along horizontal axis
# variance_h = pic_avg_h.var(axis=1)
# print variance_h

scale = 1 / 100.
print scale
# print (h/10)
#
# Plot graph for vertical axis
x_val = range(projection_v.shape[0])
y_val = projection_v*scale
# print x_val
x_val_sm = np.array(x_val)
y_val_sm = np.array(y_val)

# for smoother graph
new_x_val = np.linspace(x_val_sm.min(), x_val_sm.max(), 200)
new_y_val = spline(x_val, y_val, new_x_val)

# print new_x_val
# print new_y_val

# errorbar(x_val, y_val)
# errorbar(new_x_val, new_y_val)
# plt.fill(new_x_val, new_y_val, 'b')
# plt.plot(new_x_val, new_y_val)

# errorbar(x_val, projection_v*scale, yerr=variance_v*scale)
# print projection_v*scale
# print variance_v*scale
# imshow(pic, origin='upper', alpha=.8)

fig, ax = plt.subplots()
ax.fill_between(new_x_val, 0, new_y_val)
# ax.plot(new_x_val, new_y_val)

# Plot graph for horizontal axis
# Y_val = range(projection_h.shape[0])
# # print Y_val
# errorbar(projection_h, Y_val)
# # imshow(pic,origin='lower',alpha=.8)
# imshow(pic, origin='upper', alpha=.8)
# # plot(projection*scale)

# axis('tight')
plt.show()

# print max(projection_h)
# base = np.mean(projection_h)
# (m, i) = min((v, i) for i, v in enumerate(variance_v*scale))
# print (m, i)
# draw = ImageDraw.Draw(orig_img)
# draw.line((0, i, 2000, i), fill=128)
# orig_img.show()
