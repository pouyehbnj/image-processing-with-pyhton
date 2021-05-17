import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks
from PIL import Image
import os
import sys
from skimage import exposure
import skimage.io as ski
from skimage.transform import rescale

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

from skimage.util import random_noise

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


# rescaling images
images_rescaled = []
count = 0 

for image in images:
    new_image = rescale(image, 0.3, anti_aliasing=False)
    images_rescaled.append(new_image)
    ski.imsave(f'./scaling/{file_names[count]}',new_image)
    count =  count + 1 
    

print(images_rescaled)


#adjusting contrast
count = 0 
for image in images: 
    logarithmic_corrected = exposure.adjust_log(image, 1)
    ski.imsave(f'./contrast/raw-image-{file_names[count]}',logarithmic_corrected)
    count =  count + 1 

count = 0
for image in images_rescaled: 
    logarithmic_corrected = exposure.adjust_log(image, 1)
    ski.imsave(f'./contrast/rescaled-image-{file_names[count]}',logarithmic_corrected)
    count =  count + 1 


## denoising 
patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
count = 0 
for image in images: 
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    denoise2_fast = denoise_nl_means(image, h=0.6 * sigma_est, sigma=sigma_est,fast_mode=True, **patch_kw)
    ski.imsave(f'./denoising/raw-image-{file_names[count]}',denoise2_fast)
    count =  count + 1 

count = 0
for image in images_rescaled: 
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    denoise2_fast = denoise_nl_means(image, h=0.6 * sigma_est, sigma=sigma_est,fast_mode=True, **patch_kw)
    ski.imsave(f'./denoising/rescaled-image-{file_names[count]}',denoise2_fast)
    count =  count + 1 
    

