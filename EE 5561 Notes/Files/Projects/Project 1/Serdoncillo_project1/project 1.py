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
import matplotlib.pyplot as plt
import matplotlib.image as img
from tkinter import *
from exemplar import Exemplar

# %%
if __name__ == "__main__":
    
    # Basic test case
    bw = Exemplar()
    bw.inputImage("images/BlackWhite.png")
    bw.chooseTarget([100,140,100,140]) 
    bw.chooseSource()
    bw.showImage()
    bw.showTarget()
    bw.showSource()
    bw.run()
    
    # Mask Drawing
    bw1 = Exemplar(targetType="Free Draw", view=True)
    bw1.inputImage("images/BlackWhite.png")
    bw1.chooseTarget()
    bw1.chooseSource()
    bw1.showImage()
    bw1.showTarget()
    bw1.showSource()
    bw1.run()
    
    # Mask Drawing on Personal Butterfly Image
    bf = Exemplar(targetType="Free Draw")
    bf.inputImage("images/buttery.png")
    bf.chooseTarget()
    bf.chooseSource()
    bf.showImage()
    bf.showTarget()
    bf.showSource()
    bf.run()
    
    
    
    

    
    
    
