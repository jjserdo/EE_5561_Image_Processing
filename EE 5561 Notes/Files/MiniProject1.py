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
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy

# %% Read Image
def readImage(file):
    raw = img.imread(file)   
    image = np.asarray(raw)
    print(image.dtype)
    print(image.shape)
    print(f"Image had dimensions {image.shape[0]} by {image.shape[1]}")
    
    # Visualize image
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Original Image")
    ax.imshow(image)
    ax.set_axis_off()
    
    return image

# %% Get Target Region
def getTarget(image):
    x1 = int(input("X - Coordinate of top left point: "))
    y1 = int(input("Y - Coordinate of top left point: "))
    x2 = int(input("X - Coordinate of bottom right point: "))
    y2 = int(input("Y - Coordinate of bottom right point: "))
    target = np.full_like(image, 0)
    source = np.full_like(image, 0)
    target[x1:x2, y1:y2, :] = image[x1:x2, y1:y2, :]
    source = image - target
    
    # Visualize target and source
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Target Image")
    ax.imshow(target)
    ax.set_axis_off()
    
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Source Image")
    ax.imshow(source)
    ax.set_axis_off()
    
    return target, source

# %% Misc functions
def computeContour():
    """
    calculates the (x,y) coordinates of all p surrounding the target

    Returns
    -------
    contour list - 2 by len(contour) which is all of the x y coordinates of the contour
    """
    pass

def computePriorities():
    """
    computes the priorities at every point p on the contour
    
    Needs
    -------
    image
    contours
    patch size
    
    Returns
    -------
    priorities - array of size len(contour)
    """
    pass

def findPatch():
    """
    finds what patch to copy
    
    Needs
    -------
    source
    p
    patch size

    Returns
    -------
    finds x and y location in the source

    """
    pass


# %% Exemplar
def exemplar(file):
    IMAGE = readImage(file)
    TARGET, SOURCE = getTarget(IMAGE)
    """
    ite = 0
    # Repeat until done
    ## Identify the fill front.
    pass
    if target != 0:
        computePriorities()
        findPatch()
            ### p == np.argmax(priorities)
            ### fine exemplar 
            ### copy image data
            ### update C
    """
    
# %% Main Function
if __name__ == "__main__":
    inputFile = "fountain_girl.png"
    outputFile = exemplar(inputFile)