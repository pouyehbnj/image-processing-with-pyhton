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
from skimage import color
import skimage.transform
from skimage.transform import rescale
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
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



# rescaling images
images_rescaled = []


def rescale_images(images,path):
    count = 0 
    rescaled = []
    for image in images:
        new_image = rescale(image, 0.3, anti_aliasing=False)
        rescaled.append(new_image)
        ski.imsave(f'{path}-{file_names[count]}',new_image)
        count =  count + 1 
    return rescaled

# count = 0 
# rescaled = []
# for image in images:
#     new_image = rescale(image, 0.3, anti_aliasing=False)
#     rescaled.append(new_image)
#     ski.imsave(f'./scaling/raw-image-v1-{file_names[count]}',new_image)
#     count =  count + 1 
    
    
images_rescaled = rescale_images(images, './scaling/raw-image-v1')   

print(images_rescaled)

images_edited_version1 = []
images_rescaled_edited_version1 = []
#adjusting contrast


def adjusting_contrast(images,path):
    contrast_adjusted = []
    count = 0 
    for image in images: 
        logarithmic_corrected = exposure.adjust_log(image, 1)
        contrast_adjusted.append(logarithmic_corrected)
        ski.imsave(f'{path}-{file_names[count]}',logarithmic_corrected)
        count =  count + 1
    return contrast_adjusted         

images_edited_version1 = adjusting_contrast(images,'./contrast/raw-image-v1') 
images_rescaled_edited_version1 = adjusting_contrast(images_rescaled, './contrast/rescaled-image-v1')


## denoising 

def denoising(images,path):
    denoised = []
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    count = 0 
    for image in images: 
        sigma_est = np.mean(estimate_sigma(image, multichannel=True))
        denoise2_fast = denoise_nl_means(image, h=0.6 * sigma_est, sigma=sigma_est,fast_mode=True, **patch_kw)
        ski.imsave(f'{path}-{file_names[count]}',denoise2_fast)
        denoised.append(denoise2_fast)
        count =  count + 1
    return denoised

images_edited_version1 = denoising(images_edited_version1,'./denoising/raw-image-v1') 
images_rescaled_edited_version1 = denoising(images_rescaled_edited_version1, './denoising/rescaled-image-v1')
    

## thresholding 

def thresholding(images,path):
    count = 0 
    thresholded = []
    for image in images: 
        binary_global = image > threshold_otsu(image)
        ski.imsave(f'{path}-{file_names[count]}',binary_global)
        thresholded.append(binary_global)
        count =  count + 1
    return thresholded 


images_edited_version1 = thresholding(images_edited_version1,'./thresholding/raw-image-v1') 
images_rescaled_edited_version1 = thresholding(images_rescaled_edited_version1, './thresholding/rescaled-image-v1')

## second version of editing

images_edited_version2 = []
images_rescaled_edited_version2 = []


images_edited_version2 = denoising(images,'./denoising/raw-image-v2') 
images_edited_version2 = adjusting_contrast(images_edited_version2,'./contrast/raw-image-v2') 
images_edited_version2 = thresholding(images_edited_version2,'./thresholding/raw-image-v2') 
images_edited_version2 = rescale_images(images_edited_version2, './scaling/raw-image-v2')