import random
import sys
import random
import math
import json
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage.filters import rank_filter
from matplotlib import pyplot as plt

###################### Function to clear extra border ###################################
######### Function to put vertices in anti-clockwise order from TOP-LEFT ###########
def rectify(h):
    # this function put vertices of square we got, in anti-clockwise order
    h = h.reshape((len(h),2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

############### Main command in CLEAR_BORDER Function ###########################
def clear_border(gray):

    # remove some noises
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.bilateralFilter(gray,9, 50, 50)


    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    ret3,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plt.subplot(223),plt.imshow(thresh)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # biggest = None
    max_area = 0
    best_cnt = None
    # find image area
    image_area = gray.size
    approx = 0
    for i in contours:
        area = cv2.contourArea(i)
        if  area > image_area/4:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            # if area > max_area and len(approx)==4:
            #         biggest = approx
            #         max_area = area


    # cv2.drawContours(img,approx,-1,(0,255,0),10)
    # cv2.drawContours(img,[approx],0,(0,255,0),5,cv2.CV_AA)
    # plt.subplot(224),plt.imshow(img)

    # this is corners of new square image taken in CW order from top-left corner
    h = np.array([ [0,0],[height-1,0],[height-1,width-1],[0,width-1] ],np.float32)
    # we put the corners of biggest square in CW order to match with h
    approx=rectify(approx)
    # apply perspective transformation
    retval = cv2.getPerspectiveTransform(approx,h)
    # Now we get perfect square with size according to image size (hav to fix)
    warp = cv2.warpPerspective(img,retval,(height,width))

    # covert opencv to pil image
    cv2_im = cv2.cvtColor(warp,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    return pil_im
###################################################################################

############################# Function to crop image ###################################
######################## Downscale Function ##########################################
def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.

    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im

############# Dilate Function inside Find_components Function #######################
def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)/2,:] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)/2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image

################### find_components Functions ########################################
def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours

########## Some functions inside find_optimal_components_subset Function ############
def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    return c_info

def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)

def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)

def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)

################# find_optimal_components_subset Function #########################
def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.

    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        #print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
                print '%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                        i, covered_sum, new_sum, total, remaining_frac,
                        crop_area(crop), crop_area(new_crop), area, new_area_frac,
                        f1, new_f1)
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop
######################################################################
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
##################### Main command in CROP Function #############################
def crop(pil_im, N):
    # downscale the original image
    scale, im = downscale_image(pil_im)
    # just add this two lines24/11/2015
    gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if N == 1:
        #  for Clear_border image
        # edges = auto_canny(blurred)
        edges = cv2.Canny(np.asarray(im), 80, 150)
    else:
    # for not Clear_border image
        edges = auto_canny(blurred)
        # edges = cv2.Canny(blurred, 100, 200)
        # edges = cv2.Canny(np.asarray(im), 100, 200)

    # plt.subplot(222),plt.imshow(edges, 'gray')
    contours = find_components(edges)

    crop = find_optimal_components_subset(contours, edges)

    # upscale to the original image size.
    crop = [int(x / scale) for x in crop]
    # cropped image
    text_im = pil_im.crop(crop)

    return text_im

#################################################################################

#################################################################################
######################### Start main function ###################################
orig_img = Image.open('h.jpg')
r,g,b = orig_img.getpixel((0,0))
# convert PIL image to opencv image
img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
width, height, ch = img.shape
print width, height
# convert image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# covert opencv to pil image
gray_im = Image.fromarray(gray)

color1 =gray_im.getpixel((0,0))
color2 =gray_im.getpixel((0,width-1))
# color3 =gray_im.getpixel((0,height))
n = 0

print r,g,b
print color1
print color2
# print color3
# if pixel(0,0) is not dark color, then dont go to clear function
if color1 < 76 or color2 < 76  :
    # Call Clear_border function
    # orig_img = clear_border(gray)
    n = 1
    print n

# call Crop Function
crop_img = crop(orig_img, n)
# save cropped image
crop_img.save('crop.jpg')

plt.subplot(221),plt.imshow(img)
plt.title('Original Image')
plt.subplot(222),plt.imshow(orig_img)
plt.title('Extracted extra region Image')
plt.subplot(223),plt.imshow(crop_img)
plt.title('Cropped Image')
plt.show()
