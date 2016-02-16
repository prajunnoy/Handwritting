import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img_path = os.path.join('c:\\', 'Users','ssitang','PycharmProjects','HandwritingProj','final samples','fern data')
print img_path

img = cv2.imread(os.path.join(img_path, 'fern20.jpg'), 0)

# blur = cv2.GaussianBlur(img,(5,5),0)
thresh = cv2.adaptiveThreshold(img,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
idx=0
firstTuple_value = []
tuples = []
data_tuples = [tuples]
# find the position box of each character
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        tuples = [(leftmost),(rightmost),(topmost),(bottommost),]
        data_tuples.append(tuples)
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        # print (leftmost)
        # print (rightmost)
        # print (topmost)
        # print (bottommost)
        # print (box)

# delete first element (it detect overall image that has same size as original one)
del(data_tuples[0])
# sort data according to x-axis coordinate
sorted_data = sorted(data_tuples, key=lambda c:c[0][0])
print 'sorted data:\n', (sorted_data)

###################################### get rid of inside ########################################
insideIndex = []
flag = 1
# use loop untill there is no insideIndex (small insided letter)
while True:
    if flag == 1:
        #we dont check the first round so start with index 1
        for i in range(1, len(sorted_data)):
            if sorted_data[i-1][0][0]<sorted_data[i][0][0] and sorted_data[i][0][0]<sorted_data[i-1][1][0] and sorted_data[i][1][0]<sorted_data[i-1][1][0]:
                insideIndex.append(i)
        print 'Delete data in tuples:\n', (insideIndex)
        if insideIndex:
            for j in reversed(range(len(insideIndex))):
                del sorted_data[insideIndex[j]]
            print 'New Sorted Data:\n', (sorted_data)
            flag = 1
            insideIndex = []
        else:
            break

###################################### Cut each charactors ################################
for (x1,_),(x2,_),(_,y1),(_,y2) in sorted_data:
    idx += 1
    roi=img[y1-1:y2+1,x1-1:x2+1]
    # cv2.imwrite(str(idx) + '.jpg', roi)
    cv2.imwrite(os.path.join(img_path, './20A', str(idx) + '.jpg'), roi)
print('done!')
