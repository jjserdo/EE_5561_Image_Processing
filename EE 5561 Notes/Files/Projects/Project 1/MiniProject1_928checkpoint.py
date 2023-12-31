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
Wants:
    - [ ] Draw target and source
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy
import cv2
from skimage import io, color

# %% Read Image
def readImage(file):
    raw = img.imread(file)   
    image = np.asarray(raw)
    print(image.dtype)
    print(image.shape)
    print(f"Image had dimensions {image.shape[0]} by {image.shape[1]} by {image.shape[2]}")
    
    # Visualize image
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Original Image")
    ax.imshow(image)
    ax.set_axis_off()
    
    return image

# %% Get Target Region
def getTarget_Rectangle(image):
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

def probs(target, image):
    xx, yy, ch = image.shape
    probs = np.ones((xx, yy))
    probs[target == 0] = 0 
    
    
# %% Misc functions
def computeContour(target, probs):
    """
    calculates the (x,y) coordinates of all p surrounding the target
    >>>> not sure how to use the target array here
    Returns
    -------
    contour list - 2 by len(contour) which is all of the x y coordinates of the contour
    """
    contours = np.array([])
    # how to get the edges?
    # contour of probabilities
    laplace = cv2.Laplacian(probs)
    laplacian = laplace > 0.5
    pX = np.nonzero(laplacian)
    pY = np.nonzero(laplacian)
    p = np.vstack(pX, pY).T
    
    return p
    

def computePriorities(contours, p, target, nn=3):
    """
    computes the priorities at every point p on the contour
    >>> this assumes patch size is 3x3
    Needs
    -------
    image
    contours
    patch size
    
    Returns
    -------
    priorities - array of size len(contour),
    """
    n = int(0.5*(nn-1))
    alpha = 255 # unsure
    con = np.zeros(p.shape[0])
    dat = np.zeros(p.shape[0])
    for i in range(p.shape[1]):
        con[i] = np.sum(target[p[1,i]-n:p[1,i]+n, p[2,i]-n:p[2,i]+n]) / nn**2
        gradP = 0 # unsure
        nP =  np.array([[p[1,i+1]-p[1,i-1]],[p[2,i+1]-p[2,i-1]]])
        nP /= nP / np.linalg.norm(nP)
        dat[i] = np.abs(np.dot(gradP, np))/alpha

    priorities = con * dat
    patchMe = np.argmax(priorities)
    
    return con, data, patchMe
    
def findPatch(patchMe, target, source):
    """
    finds what patch to copy
    
    Needs
    -------
    source
    p
    patch size
    >> better have patch tbh

    Returns
    -------
    finds x and y location in the source

    """
    
    slab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    plab = cv2.cvtColor(patch , cv2.COLOR_BGR2LAB)
    best = np.zeros((n,n))
    #sLab = color.rgb2lab(rgb)
    for pp in slab != 0:
        dist = patch - slab[pp-n:pp+n,pp-n:pp+n]
        current = np.sum(dist * dist)
        if current <= best:
            best = np.copy(current)
    bestRGB = cv2.cvtColor(best, cv2.COLOR_LAB2BGR)
    target[pp-n:pp+n,pp-n:pp+n] += bestRGB
    
    return target


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
    
# %% Exemplar as a class
class Exemplar():
    def __init__(self, patchSize = 3, targetType = "Rectangle", sourceType = "All"):
        self.n = patchSize
        self.tt = targetType
        self.ss = sourceType
        
    def inputImage(self, imageFile):
        raw = img.imread(imageFile)   
        image = np.asarray(raw)
        self.input = image
        self.shape = image.shape
        
    def chooseTarget(self):
        gucci = True
        if self.tt == "Rectangle":
            x1 = int(input("X - Coordinate of top left point: "))
            y1 = int(input("Y - Coordinate of top left point: "))
            x2 = int(input("X - Coordinate of bottom right point: "))
            y2 = int(input("Y - Coordinate of bottom right point: "))
            target = np.full_like(self.image, 0)
            source = np.full_like(self.image, 0)
            target[x1:x2, y1:y2, :] = self.image[x1:x2, y1:y2, :]
            source = self.image - target
            
            # Visualize target and source
            fig, ax = plt.subplots(dpi=300)
            ax.set_title("Target Image")
            ax.imshow(target)
            ax.set_axis_off()
            
            fig, ax = plt.subplots(dpi=300)
            ax.set_title("Source Image")
            ax.imshow(source)
            ax.set_axis_off()
            
            self.target = target
            self.source = source
        elif self.tt == "Polygon Draw":
            pass
        elif self.tt == "Free Draw":
            pass
        else:
            gucci = False
        self.target_is_chosen = np.copy(gucci)
            
    
    def chooseSource(self):
        gucci = True
        if self.target_is_chose:
            if self.cc == "Padding":
                pass
            elif self.cc == "Rectangle":
                pass
            elif self.cc == "All":
                pass
            elif self.cc == "Free Draw":
                pass
            else:
                gucci = False
        else:
            gucci = False
                
        self.source_is_chosen = np.copy(gucci)
        
    def letsGo(self):
        if self.target_is_chosen and self.source_is_chosen:
            self.update()
        else:
            print(f"target: {self.target_is_chosen} \n source: {self.source_is_chosen}")
        
    def update(): # Repeat until done:
        # 1a Identify the fill front
        
        # 1b Compute priorities of all P in the contour of target
        
        # 2a Find patch with maximum priority
        interestedPath = np.unravel_index(np.argmax(priorities), np.array(priorities).shape)
        # 2b Find the exemplar source that minimizes SSD
        
        # 2c Copy image data from exemplar to the patch
        
        # 
    
# %% Main Function
if __name__ == "__main__":
    inputFile = "fountain_girl.png"
    outputFile = exemplar(inputFile)
    
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:11:31 2023

@author: justi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy
import cv2

def test():
    file = "fountain_girl.png"
    raw = img.imread(file)   
    image = np.asarray(raw)
    print(image.dtype)
    print(image.shape)
    x1 = int(input("X - Coordinate of top left point: "))
    y1 = int(input("Y - Coordinate of top left point: "))
    x2 = int(input("X - Coordinate of bottom right point: "))
    y2 = int(input("Y - Coordinate of bottom right point: "))
    #target = np.full_like(image, 0)
    #source = np.full_like(image, 0)
    
# %%
# ============================================================================

CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.cv.SetMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = np.zeros(CANVAS_SIZE, np.uint8)
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

# ============================================================================
"""
if __name__ == "__main__":
    pd = PolygonDrawer("Polygon")
    image = pd.run()
    cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)
"""

bw = np.zeros((256,256))
bw[:50,:] = 128
bw[50:200,:] = 255
cv2.imwrite("blackWhite.png", bw)
