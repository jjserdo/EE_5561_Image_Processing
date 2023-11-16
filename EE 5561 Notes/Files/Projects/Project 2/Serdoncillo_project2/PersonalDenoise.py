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
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

# %% jjGausian class

class jjGausian():
    pass

# %% jjMedian class

class jjMedian():
    pass

# %% jjPGD and jjADMM class
# Blur Kernel
ksize = 9 
kernel = np.ones((ksize,ksize)) / ksize**2

[h, w] = I.shape
kernelimage = np.zeros((h,w))
kernelimage[0:ksize, 0:ksize] = np.copy(kernel)
fftkernel = np.fft.fft2(kernelimage)

sigm = np.sqrt(0.1)
alpha = np.sqrt(sigm**2/ np.max(np.abs(fftkernel)))

def H(x):
    return np.real(np.fft.ifft2(np.fft.fft2(x) * fftkernel))

def HT(x):
    return np.real(np.fft.ifft2(np.fft.fft2(x) * np.conj(fftkernel)))

def A(x):
    return HT(H(x)) + rho*x

class jjPGD():
    step_size = alpha**2/sigm**2
    # JJ's code starts here 
    x = np.copy(y)
    aTV = 1e-2
    save = [500, 1000, 2500, 5000]
    saves = []
    for ind in range(7500):
        x_n = x + step_size * HT(y - H(x))
        x = TV_denoise(x_n, weight=aTV)
        if ind in save:
            saves.append(x)
            
class jjADMM():
    z = np.zeros_like(y)
    u = np.zeros_like(y)
    aTV = 1e-2
    rho = 0.1
    no_iter=20
    save = [50, 100, 250]
    saves = []
    x = np.copy(z)
    for ind in range(500):
        b = HT(y) + rho*(z-u)
        
        # cg_solve equivalent
        x0 = np.zeros_like(b)
        xcg = np.copy(x0)
        r = b - A(xcg)
        p = np.copy(r)
        rsold = np.sum(r * np.conj(r))
        for i in range(no_iter):
            Ap = A(p)
            alpha = rsold/np.sum(p * Ap)
            xcg = xcg + alpha * p
            r = r - alpha * Ap
            rsnew = np.sum(r * np.conj(r))
            p = r + rsnew / rsold * p
            rsold = np.copy(rsnew)
        x = xcg.copy()
        
        z = TV_denoise(x + u, weight=aTV)
        u += x - z
        if ind in save:
            saves.append(x)