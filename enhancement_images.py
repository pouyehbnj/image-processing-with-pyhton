import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks
from PIL import Image
import os
import sys
from skimage import color
import skimage.io as ski
from skimage.transform import rescale, resize, downscale_local_mean

file_names =[]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        file_names.append(filename)
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder('./images')
print(images)
# plt.figure(figsize=(4, 4))
# plt.imshow(images[0], cmap='gray')
# plt.axis('off')
# plt.show()


# rescalong images
images_rescaled = []
count = 0 

for image in images:
    new_image = rescale(image, 0.3, anti_aliasing=False)
    images_rescaled.append(new_image)
    ski.imsave(f'./scaling/{file_names[count]}',new_image)
    count =  count + 1 
    

print(images_rescaled)
    
