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
    def __init__(self, image, h=10, searchWindow=21, squareNeighborhood=7):
        self.h = h # 10 * standard deviation of 1 
        self.k = searchWindow
        self.n = squareNeighborhood
        self.image = image
        self.row, self.col = image.shape
        
        self.n2 = int(0.5*(self.n-1))
        self.k2 = int(0.5*(self.k-1))
        
    def run(self):
        what = self.h**2 * self.n**2
        output = np.zeros_like(self.image)
        
        pad = np.pad(self.image, self.k2, mode='reflect')
        pr, pc = pad.shape
        neigh = np.zeros((pr, pc, self.n, self.k))
        
        for ii in range(self.k2, self.k2 + self.row):
            for jj in range(self.k2, self.k2 + self.col):
                neigh[ii,jj] = pad[ii-self.n2:ii+self.n2+1, jj-self.n2:jj+self.n2+1]
                
        for ii in range(self.k2, self.k2 + self.row):
            for jj in range(self.k2, self.k2 + self.col):
                hehe = neigh[ii,jj]
                hihi = neigh[ii-self.k2:ii+self.k2+1, jj-self.k2:jj+self.k2+1]
                r, c = hehe.shape
                er, ec = int(0.5*(r-1)), int(0.5*(r-1))
                hr, hc = hihi.shape
                
                numer = 0
                denum = 0
                
                for aa in range(hr):
                    for bb in range(hc):
                        qq = hihi[aa,bb]
                        rr = qq[er, ec]
                        normNorm = np.exp(-1 * ((np.sum((hehe-qq)**2))/what))
                        numer += normNorm * rr
                        denum += normNorm
                
                Ip = numer/denum
                
                output[ii-self.k2,jj-self.k2] = max(min(255, Ip), 0)
                
        self.output = np.copy(output)
                
        fig, ax = plt.subplots(1,2, figsize=(6,6), dpi=150)
        ax[0].imshow(self.image, cmap="gray")
        ax[0].set_title("Before")
        ax[0].axis('off')
        ax[1].imshow(self.output, cmap="gray")
        ax[1].set_title("After")
        ax[1].axis('off')
        fig.tight_layout()
                
        return output
    
    def metrics(self):
        pass
        
        
        

       
