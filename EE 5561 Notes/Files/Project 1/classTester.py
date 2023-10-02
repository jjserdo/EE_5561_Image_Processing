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


import numpy as np
import cv2
#from MiniProject1 import Exemplar

"""
Image Creation
"""
def createImages():
    bw = np.zeros((256,256))
    bw[:128,:] = 100
    bw[128:,:] = 200
    cv2.imwrite("images/blackWhite.png", bw)
    

""" 
Image Testing
"""
if __name__ == "__main__":
    
    #createImages()
    f = "images/blackWhite.png"
    """
    black = Exemplar()
    black.inputImage(f)
    black.showInput()
    black.chooseSource()
    """
    image = cv2.imread(f)
    print(imread.shape)
    img2 = cv2.Laplacian(image, ddepth)
