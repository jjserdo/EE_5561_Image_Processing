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
from MiniProject1_v2 import Exemplar

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
    
def createImages():
    bw = np.zeros((200,200))
    bw[:128,:] = 100
    bw[128:,:] = 200
    cv2.imwrite("images/blackWhite.png", bw)



# %%
if __name__ == "__main__":
    """
    fgBW = Exemplar()
    fgBW.inputImage("images/bw_fountain_girl.png")
        # fgBW shape (929, 695, 3)
    #fgBW.showInput()
    
    #### Target Options
    #fgBW.chooseTargetClick()
    #fgBW.chooseTarget()
    fgBW.chooseTarget([495,870,199,486])
    fgBW.showTarget(1)
    """
    #createImages()
    
    # Black and White Image
    """
    bw = Exemplar(patchSize=9, targetType="Free Draw", view=False)
    bw.inputImage("images/BlackWhite.png")
    #bw.inputImage("images/images/doggy1.png")
    #bw.chooseTarget([100,140,100,140]) # Basic test case
    bw.chooseTarget() 
    bw.chooseSource()
    #bw.showImage()
    #bw.showTarget()
    #bw.showSource()
    bw.run()
    #bw.run(4) 
    #bw.run(2) 
    """
    # Fountain Girl
    fg = Exemplar(patchSize=9, targetType="Free Draw", view=True) #Rectangle Coords")
    fg.inputImage("images/spider.png")
    #fg.chooseTarget([30,100,100,170])
    #fg.chooseTarget([100,160,60,120])
    fg.chooseTarget()
    fg.chooseSource()
    #fg.showImage()
    #fg.showTarget()
    #fg.showSource()
    #fg.run(2)
    #fg.run()  # Ite 80 1039-1040 200x200 40x40 [120,160,60,100]
    #fg.run()  # Ite 80 1042-1040 200x200 40x40 [60,100,60,100]
    #fg.run() # Ite 285 1045-1049 "" patchSize=5
    fg.run(0)
    # 9 - 176, 5 - , 13 - 
    
    cv2.imwrite("images/images/spiderRaw.png", fg.evolve0)
    cv2.imwrite("images/images/spider.png", fg.evolve)
    
    """
    tri = Exemplar(targetType="Free Draw")
    tri.inputImage("images/images/doggy1.png")
    tri.chooseTarget()
    tri.chooseSource()
    tri.run()
    """
    #cv2.imwrite("images/images/bw1Raw.png", bw.evolve0)
    #cv2.imwrite("images/images/bw1.png", bw.evolve)
    
    

    
    
    
