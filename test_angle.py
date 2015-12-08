import Image
from pylab import *
from scipy import *
from scipy.optimize import leastsq

# Load the picture with PIL, process if needed
pic = asarray(Image.open("final.jpg"))

# Average the pixel values along horizontal axis
pic_avg_h = pic.mean(axis=1)
# print pic_avg_h
projection_h = pic_avg_h.sum(axis=1)
print projection_h

# Compute the variance along horizontal axis
variance_h = pic_avg_h.var(axis=1)
# print variance_h

# Average the pixel values along vertical axis
# pic_avg_v = pic.mean(axis=2)
# projection_v = pic_avg_v.sum(axis=0)

# Compute the variance along vertical axis
# variance_v = pic_avg_v.var(axis=0)

scale = 1/40.

# Plot graph for vertical axis
# X_val = range(projection_v.shape[0])
# errorbar(X_val, projection_v*scale, yerr=variance_v*scale)
# imshow(pic, origin='lower', alpha=.8)

# Plot graph for horizontal axis
Y_val = range(projection_h.shape[0])
# print Y_val
errorbar(projection_h, Y_val)
# imshow(pic,origin='lower',alpha=.8)
imshow(pic, origin='upper', alpha=.8)
# plot(projection*scale)

axis('tight')
show()