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

# %% Importing packages and functions

import numpy as np 
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import imageio as iio

# %% jjGaussNoise class

class jjGaussNoise():
    def __init__(self, mean=0, var=25):
        self.mean = mean
        self.var  = var

    def noise(self, image):
        row, col, _  = image.shape
        gauss  = np.ceil(np.random.normal(self.mean, self.var**0.5, (row,col))).astype(np.int16)
        gauss = np.repeat(gauss[:, :, np.newaxis], 3, axis=2)
        output = image + gauss
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output

# %% jjSnP class

class jjSnP():
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.sp = salt_prob
        self.pp = pepper_prob

    def noise(self, image):
        output = np.copy(image)
        row, col, _  = image.shape
        num_salt = int(np.ceil(self.sp * row * col))
        for i in range(num_salt):
            c = [np.random.randint(0, i-1) for i in image.shape]
            output[c[0], c[1], :] = 255
        num_pepper = int(np.ceil(self.pp * row * col))
        for i in range(num_pepper):    
            c = [np.random.randint(0, i-1) for i in image.shape]
            output[c[0], c[1], :] = 0

        return output.astype(np.uint8)
