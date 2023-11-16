
'''
Justine Serdoncillo
EE 5561 - Image Processing
Problem Set 3
November 7, 2023
'''

# %% Problem Statement
"""
    2) [9 pts] Programming Exercise 1: In this exercise, you will familiarize yourself with
    function handles. Write function handles to calculate the following:
    a) The image derivative in x direction (the built-in function numpy.diff(Python)/diff(MATLAB)
    may be useful),
    b) the image derivative in y direction,
    c) the magnitude of the gradient vector in (x, y) directions,
    d) discrete cosine transform (you may use the scipy.fftpack.dct(Python)/dct2(MATLAB)
    function for this) of each 8 × 8 distinct block in the image.
    For a, b, and c make the derivative operator circulant. Thus, the output should be the same
    size as the image. Verify all 4 function handles on the cameraman image.
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy
import imageio as iio
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d

# %% Function Handles
def gradX(image):   
    shift = np.roll(image, 1, axis=1)
    x = image - shift
    return x

def gradY(image):
    shift = np.roll(image, 1, axis=0)
    y = image - shift
    return y

def gradMag(image):
    x = gradX(image)
    y = gradY(image)
    mag = np.sqrt(x**2 + y**2)
    return mag

def dct2c(img):
    return dct(dct(img, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

def DCT(image, size):
    [Nx,Ny] = image.shape
    dct = np.zeros_like(image)
    for x in range(0,Nx,size):
        for y in range(0,Ny,size):
            dct[x:x+8,y:y+8] = dct2c(image[x:x+8,y:y+8])
    return dct


# %% Importing Images
man_img  = np.complex64(iio.imread('cameraman.tif'))

# %% Problem 2
def prob2():
    print(f"Cameraman has size {man_img.shape}")
    x = gradX(man_img)
    y = gradY(man_img)
    M = gradMag(man_img)
    D = DCT(man_img, 8)
    print(f"Derivatives has size {x.shape}")
    
    # Plot the figures
    fig, ax = plt.subplots(2,2, figsize=(8,8), dpi=150)
    ax[0,0].imshow(np.abs(man_img), cmap="gray")
    ax[0,0].set_title("Original Image")
    ax[0,1].imshow(np.abs(x), cmap="gray")
    ax[0,1].set_title("Image x-derivative")
    ax[1,0].imshow(np.abs(y), cmap="gray")
    ax[1,0].set_title("Image y-derivative")
    ax[1,1].imshow(np.abs(M), cmap="gray")
    ax[1,1].set_title("Image derivative magnitude")
    
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    ax.imshow(np.abs(D), cmap="gray")
    ax.set_title("DCT Image")
    
# %% Main Function
if __name__ == "__main__":
    prob2()
    
    

    
    

    
    