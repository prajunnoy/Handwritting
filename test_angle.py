import Image
import collections
from pylab import *
from scipy import *
from scipy.optimize import leastsq

# Load the picture with PIL, process if needed
orig_img = Image.open("rotate.jpg")
pic = asarray(orig_img)

# imshow(pic)
# show()

# Average the pixel values along horizontal axis
pic_avg_h = pic.mean(axis=1)
# print pic_avg_h
projection_h = pic_avg_h.sum(axis=1)
print projection_h
# print len(projection_h)

# Compute the variance along horizontal axis
variance_h = pic_avg_h.var(axis=1)
# print variance_h

# Average the pixel values along vertical axis
# pic_avg_v = pic.mean(axis=2)
# projection_v = pic_avg_v.sum(axis=0)

# Compute the variance along vertical axis
# variance_v = pic_avg_v.var(axis=0)

scale = 1 / 40.

##############################################
# segmentation to each sentence
y_int = 0
prev_i = 0
index = []
proj_num = []
count = 0
initial = 0
for i in projection_h:
    if i < 765 and prev_i == 0:
        initial = 0
    elif prev_i < 765 and i == 765 and initial == 0:
        initial = 1
    # there is space between edge and text
    elif (i < 765 and prev_i == 765)or(prev_i < 765 and i == 765 and prev_i != 0 and initial == 1):
    # no space between edge and text
    # elif (i < 764 and (764 < prev_i <= 765))or(prev_i < 764 and (764 < i <= 765) and prev_i != 0 and initial == 1):
    # elif (765 > i > 764 and (764 < prev_i <= 765))or(765 > prev_i > 764 and (764 < i <= 765) and prev_i != 0 and initial == 1):
    # elif i < 765 and prev_i == 765 and initial == 1:
        index.append(count)
        proj_num.append(y_int)
    elif (764 < i < 765 and 764 < prev_i < 765)or(764 < prev_i < 765 and 764 < i < 765 and prev_i != 0 and initial == 1):
        count -= 1
        index.append(count)
        proj_num.append(y_int)
    y_int += 1
    prev_i = i
    count += 1
    # print initial
print index
print proj_num
print len(index)
print orig_img.size

width, height = orig_img.size

# crop image for old value of index eg. [1,3,4,..] y1=1 y2=3
# for y1, y2 in zip(index, index[1:])[::2]:
#     if y2 - y1 > 40:
#         c += 1
#         box = (0, y1, width, y2)
#         print y1, y2
#         crop_img = orig_img.crop(box)
#         crop_img.save('sentence'+str(c)+'.jpg')
# print c

# count frequency of element in list
counter = collections.Counter(index)
c_frq  = (counter.items())
c_frq = sorted(c_frq, key=lambda tup: tup[0])
print (c_frq)

flag = 0
start = 0
c = 0
n1 = 0
n2 = 0
# add some more condition to loop more value for better sentence
for _, n in sorted(counter.items(), key=lambda tup: tup[0]):
    if n1==0 and n2 == 0:
        n1 = n
    elif (n2==0)or(n1!=0 and n2!=0):
        if n1!=0 and n2!=0:
            n1 = n2
            start -= n1
        n2 = n
        for y1 in proj_num[start:start+1:]:
            start += n1+n2
            for y2 in proj_num[(start)-1::]:
                c += 1

                if y2-y1 >30:
                    box = (0, y1, width, y2)
                    print y1, y2
                    crop_img = orig_img.crop(box)
                    crop_img.save('sentence'+str(c)+'.jpg')
                break

print c
# for _, n1 in c_frq:
#     for __, n2 in c_frq[::2]:
#         if n1!=0 and n2!=0:
#             for y1 in proj_num[:n1:]:
#                 for y2 in proj_num[:n2:]:


# 'enumerate' create index no. for each element in list
# print (i for i, (x, u) in enumerate(index) if index.count(x) > 1)

# example of for-loop for [start:stop:step]
# for num1, y in index[::step]:


# crop_img = orig_img.crop((0, 32, width, 354))
# imshow(crop_img)
# show()
################################################

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
