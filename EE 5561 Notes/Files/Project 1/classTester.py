'''
Justine Serdoncillo
EE 5561 - Image Processing
Mini-Project 1
October 19, 2023
'''

"""
Implementation of 
"Region Filling and Object Removal by Exemplar-Based Image Painting"
A. Criminisi, P. Perez, K. Toyama
Microsoft Research 2004
"""

""" 
Image Testing
"""
import numpy as np
import cv2
from MiniProject1 import Exemplar

def createImages():
    bw = np.zeros((256,256))
    bw[:50,:] = 128
    bw[50:200,:] = 255
    cv2.imwrite("blackWhite.png", bw)
    


if __name__ == "__main__":
    f = "images/blackWhite.png"
    black = Exemplar()
    
