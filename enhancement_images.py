import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks
from PIL import Image
import os, sys
from skimage import color


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder('./images')

# plt.figure(figsize=(4, 4))
# plt.imshow(images[0], cmap='gray')
# plt.axis('off')
# plt.show()


# size of blocks
views = []
block_shape = (3, 3)

# see astronaut as a matrix of blocks (of shape block_shape)
# for image in images:
#     views.append(view_as_blocks(image, block_shape))
#     print(views)

#!/usr/bin/python

block_shape = (4, 4)
resized_images = []
for image in images:
    resized_images.append(view_as_blocks(color.rgb2gray(image), block_shape))
     imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

skimage.io.imsave(fname="chair.tif", arr=image)



