'''
Justine Serdoncillo
EE 5561 - Image Processing
Mini-Project 2
November 9, 2023
'''

"""
Implementation of 
"A non-local algorithm for image denoising"
A. Buades, B. Coll, J.M. Morel
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
from tkinter import *
import imageio as iio
#from NonLocalDenoise import NonLocal
#from PersonalDenoise import jjGausian, jjMedian, jjPGD, jjADMM
#from PersonalNoise   import jjGaussNoise, jjSnP

# %%
if __name__ == "__main__":
    images = []
    lena_img = np.complex64(plt.imread('images/lena512.bmp'))
    man_img  = np.complex64(iio.imread('images/man.png'))
    coin_img = np.complex64(plt.imread('images/eight.tif'))
    cman_img = np.complex64(iio.imread('images/cameraman.tif'))
    images.append(lena_img)
    images.append(man_img)
    images.append(coin_img)
    images.append(cman_img)
    
    gnoise = []
    snoise = []
    for i in range(4):
        # here I will want to create the noisy images
        pass
        # Types of Noise
            # Gaussian Noise
            # Salt and Pepper Noise
    
    # Compare with other methods but just use skimage 
        # Gaussian Filter
        # Median Filter    
        # TV Denoiser PGD 
        # TV Denoiser ADMM
    
    # Metrics
        # PSNR
        # MSE
    
    
    

    
    
    
