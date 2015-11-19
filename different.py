import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from matplotlib import pyplot as plt

orig_im = Image.open('crop.jpg')
bg = Image.open('inpaint.jpg')

new = ImageChops.difference(orig_im, bg)

new.save('new.jpg')

plt.subplot(221),plt.imshow(orig_im)
# plt.title('Original Image')
plt.subplot(222),plt.imshow(bg)
# plt.title('Extracted extra region Image')
plt.subplot(223),plt.imshow(new)
# plt.title('Cropped Image')
plt.show()
