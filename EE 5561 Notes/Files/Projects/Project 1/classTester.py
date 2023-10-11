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
from MiniProject1 import Exemplar

# %% Functions Available
# inputImage, showInput
# chooseTargetClick[x], chooseTarget, showTarget
# chooseSource
# run

### background functions
# update_all
    # comp_fill_front
    # comp_priorities
        # comp_data
    # max_priority
    # find_exemplar
    # copy_image
    # update_priorities



# %%
if __name__ == "__main__":
    fgBW = Exemplar()
    fgBW.inputImage("images/bw_fountain_girl.png")
        # fgBW shape (929, 695, 3)
    #fgBW.showInput()
    
    #### Target Options
    #fgBW.chooseTargetClick()
    #fgBW.chooseTarget()
    fgBW.chooseTarget([495,870,199,486])
    fgBW.showTarget(1)
    
    

    

    
    
    
