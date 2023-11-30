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
try:
    from skimage.restoration import denoise_tv_chambolle as TV_denoise
except ImportError:
    from skimage.filters import denoise_tv_chambolle as TV_denoise


# %% jjGausian class

class jjGausian():
    def __init__(self, sigma=1):
        self.sigma = sigma

    def denoise(self, image):
        size = int(2*(np.ceil(3*self.sigma))+1)
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*self.sigma**2)))
        g = g/g.sum()

        output = convolve2d(image[:,:,0], g, mode='same', boundary='symm')
        output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
        return output.astype(np.uint8)

# %% jjMedian class

class jjMedian():
    def __init__(self, radius=1):
        self.radius = radius

    def denoise(self, image):
        padded_image = np.pad(image, self.radius, mode='edge')

        output = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j, :] = np.median(padded_image[i:i+2*self.radius+1, j:j+2*self.radius+1, :])
        return output.astype(np.uint8)

# %% jjPGD and jjADMM class
class jjPGD():
    def __init__(self, ksize=9, sigm=np.sqrt(0.1), aTV=1e-2):
        self.ksize = ksize
        self.kernel = np.ones((ksize,ksize)) / ksize**2
        self.sigm  = sigm
        self.aTV = aTV
    
    def denoise(self, image, ite=1000):
        def H(x):
            return np.real(np.fft.ifft2(np.fft.fft2(x) * fftkernel))
        def HT(x):
            return np.real(np.fft.ifft2(np.fft.fft2(x) * np.conj(fftkernel)))
        
        image = np.copy(image[:,:,0])
        row, col = image.shape
        kernelimage = np.zeros_like(image)
        kernelimage[0:self.ksize, 0:self.ksize] = np.copy(self.kernel)
        fftkernel = np.fft.fft2(kernelimage)
        alpha = np.sqrt(self.sigm**2/ np.max(np.abs(fftkernel)))
        step_size = alpha**2/self.sigm**2
        x = np.copy(image)
        for ind in range(ite):
            x_n = x + step_size * HT(image - H(x))
            x = TV_denoise(x_n, weight=self.aTV)
        x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
        return x.astype(np.uint8)
        
            
class jjADMM():
    def __init__(self, ksize=9, sigm=np.sqrt(0.1), aTV=1e-2, rho=0.1):
        self.ksize = ksize
        self.kernel = np.ones((ksize,ksize)) / ksize**2
        self.sigm  = sigm
        self.aTV = aTV
        self.rho = rho
        
    def denoise(self, image, ite=50, no_iter=15):
        def H(x):
            return np.real(np.fft.ifft2(np.fft.fft2(x) * fftkernel))
        def HT(x):
            return np.real(np.fft.ifft2(np.fft.fft2(x) * np.conj(fftkernel)))
        def A(x):
            return HT(H(x)) + self.rho*x
        image = np.copy(image[:,:,0])
        row, col = image.shape
        kernelimage = np.zeros_like(image)
        kernelimage[0:self.ksize, 0:self.ksize] = np.copy(self.kernel)
        fftkernel = np.fft.fft2(kernelimage)
        alpha = np.sqrt(self.sigm**2/ np.max(np.abs(fftkernel)))
        
        z = np.zeros_like(image).astype(np.float64)
        u = np.zeros_like(image).astype(np.float64)
        x = np.copy(image).astype(np.float64)
        for ind in range(ite):
            b = HT(image) + self.rho*(z-u)
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
            
            z = TV_denoise(x + u, weight=self.aTV)
            u += x - z
        x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
        return x.astype(np.uint8)