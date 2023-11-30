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

# %% Main NonLocal class
class NonLocal():
    def __init__(self, h=10, searchWindow=21, squareNeighborhood=7):
        self.h = h 
        self.k = searchWindow
        self.n = squareNeighborhood
        
        self.n2 = int(0.5*(self.n-1))
        self.k2 = int(0.5*(self.k-1))
        
    def denoise(self, image):
        row, col, _ = image.shape
        image = np.copy(image[:,:,0])
        output = np.zeros_like(image)
        
        pad = np.pad(image, self.k2, mode='reflect')
        pr, pc = pad.shape
        neigh = np.zeros((pr, pc, self.n, self.n))
        
        for ii in range(self.k2, self.k2 + row):
            for jj in range(self.k2, self.k2 + col):
                neigh[ii,jj,:,:] = pad[ii-self.n2:ii+self.n2+1, jj-self.n2:jj+self.n2+1]
                
        for ii in range(self.k2, self.k2 + row):
            for jj in range(self.k2, self.k2 + col):
                hehe = neigh[ii,jj,:,:]
                hihi = neigh[ii-self.k2:ii+self.k2+1, jj-self.k2:jj+self.k2+1,:,:]
                r, c = hehe.shape
                er, ec = int(0.5*(r-1)), int(0.5*(r-1))
                hr, hc, _, _ = hihi.shape
                
                numer = 0
                denum = 0
                
                for aa in range(hr):
                    for bb in range(hc):
                        qq = hihi[aa,bb]
                        rr = qq[er, ec]
                        normNorm = np.exp(-1 * ((np.sum((hehe-qq)**2))/self.h**2))
                        numer += normNorm * rr
                        denum += normNorm
                
                weight = numer/denum
                
                output[ii-self.k2,jj-self.k2] = max(min(255, weight), 0)
        output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
        return output.astype(np.uint8)
    
        
        
        

       
