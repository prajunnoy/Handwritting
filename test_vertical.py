import Image
import collections
import itertools
from pylab import *
from scipy import *
from scipy.optimize import leastsq

# Load the picture with PIL, process if needed
orig_img = Image.open("sentence22.jpg")
pic = asarray(orig_img)

# Vertical Projection
# Average the pixel values along vertical axis
pic_avg_v = pic.mean(axis=2)
projection_v = pic_avg_v.sum(axis=0)
print projection_v
# Compute the variance along vertical axis
variance_v = pic_avg_v.var(axis=0)

scale = 1 / 40.

# apply Horizontal projection Graph to crop each sentence
print max(projection_v)
max_num = max(projection_v)
prev_i = 0
count = 0
index = []
spacef = 0
spacel = 0
for i in projection_v:
    if (i < max_num and prev_i == max_num and spacef > 20):
        index.append(count)
        spacef = 0
    elif prev_i < max_num and i == max_num and prev_i != 0 and((spacel > 200 and spacef > 4)or(spacel > 60 and (12 <= spacef or spacef == 0))):
        index.append(count)
        spacel = 0
    if i == max_num and prev_i == max_num:
        spacef += 1
    elif (i < max_num and prev_i == max_num)or(i < max_num and prev_i < max_num):
        spacel +=1
    print spacef, spacel
    prev_i = i
    count += 1
print count
print index

width, height = orig_img.size

c = 0
y = 0
# for y1, y2 in itertools.combinations(index, r=2):
    # print (y1,y2)
    # if 250> y1-y > 60 or y == 0:
for y1, y2 in zip(index, index[1:])[::2]:
    c += 1
    box = (y1, 0, y2, height)
    # print y1, y2
    crop_img = orig_img.crop(box)
    crop_img.save('word'+str(c)+'.jpg')
print c

# Plot graph for vertical axis
X_val = range(projection_v.shape[0])
errorbar(X_val, projection_v*scale, yerr=variance_v*scale)
imshow(pic, origin='upper', alpha=.8)

axis('tight')
show()