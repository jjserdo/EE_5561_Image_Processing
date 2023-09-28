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
    - [ ] Draw target 
    - [ ] Draw source
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy
import cv2
from skimage import io, color
       
class Exemplar():
    def __init__(self, patchSize=9, targetType ="Rectangle", sourceType = "All"):
        # Exemplar class properties
        self.n = patchSize
        self.tt = targetType
        self.ss = sourceType
        
        # Image, Mask and Source are CONSTANT
        self.image = None
        self.mask = None
        self.source = None
        
        # Target and Probabilities are Changing
        self.evolve = None
        self.conf = None
        
        # Misc values
        self.grad = None
        self.lab = None
        self.n2 = int(0.5*(self.n-1))
        self.limit = 600
        self.count = 1
        
    def inputImage(self, imageFile):
        raw = img.imread(imageFile)   
        image = np.asarray(raw)
        self.image = image
        self.shape = image.shape
        self.count = 1
        
    def showInput(self):
        if np.shape(self.image) == ():
            print("Empty Image")
            return False
        else:
            cv2.imshow("display", self.image)
            input = cv2.waitKey(0)
            ascii_input = input % 256
            print(ascii_input)
            return True
        
    def chooseTarget(self):
        if self.image != None:
            if self.tt == "Rectangle":
                x1 = int(input("X - Coordinate of top left point: "))
                y1 = int(input("Y - Coordinate of top left point: "))
                x2 = int(input("X - Coordinate of bottom right point: "))
                y2 = int(input("Y - Coordinate of bottom right point: "))
                target = np.ones(self.image)
                target[x1:x2, y1:y2, :] = 0
    
                # Visualize target and source
                fig, ax = plt.subplots(dpi=300)
                ax.set_title("Target Image")
                ax.imshow(target)
                ax.set_axis_off()
            elif self.tt == "Polygon Draw":
                pass
            elif self.tt == "Free Draw":
                pass
        else:
            print("Get an input first!")
            return False

        # constant
        self.mask = target
        
        # changing
        self.evolve = self.image * target
        self.conf = target
        
        # misc
        self.count = 1
        return True
        
    def showTarget(self):
        if np.shape(self.image) == ():
            print("Empty Target")
            return False
        else:
            cv2.imshow("display", self.mask)
            input = cv2.waitKey(0)
            ascii_input = input % 256
            print(ascii_input)
            return True
               
    def chooseSource(self):
        if self.image != None and self.mask != None:
            if self.cc == "Padding":
                pass
            elif self.cc == "Rectangle":
                pass
            elif self.cc == "All":
                self.source = self.image * np.abs(1 - self.mask)
            elif self.cc == "Free Draw":
                pass
        else:
            print("Choose an image and target first!")
            return False
        
        self.lab = cv2.cvtColor(self.source, cv2.COLOR_BGR2LAB)
        self.count = 1
        return True
        
    def run(self):
        if self.image != None and self.mask != None and self.source != None:
            self.grad = np.zeros(1)
            while self.count < self.limit and np.min(self.conf) != 0:
                self.update()
        else:
            print("Choose image, target and source first!")
    
    def comp_prio(self,listofP):
        g = listofP.shape[0]
        confidence = np.zeros(g)
        for i in range(g):
            px = listofP[g,0]
            py = listofP[g,1]
            confidence[i] = np.sum(self.conf[px-self.n2:px+self.n2,py-self.n2:py+self.n2])
        return confidence
    
    def get_source(self,p):
        compare = cv2.cvtColor(self.evolve[p[0]-self.n2:p[0]+self.n2,p[1]-self.n2:p[1]+self.n2,:], cv2.COLOR_BGR2LAB)
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
        """
        g = length of source?
        ssd = np.zeros(2d)
        for i in range(g):
            sourceX
            sourceY
            ssd[i] = np.sum((self.lab[a-self.n2:a+self.n2,p-self.n2:p+self.n2,:]-compare)**2)
        source_patch = np.unravel_index(np.argmax(ssd))
        return source_patch
        
    def update(self): # Repeat until done:
        self.count += 1
        # 1a Identify the fill front
        fillFront()
        # 1b Compute priorities of all P in the contour of target
        confidence = self.comp_prio(ListofP)
        
        priorities = confidence * data
        # 2a Find patch with maximum priority
        p = np.unravel_index(np.argmax(priorities), np.array(priorities).shape)
        # 2b Find the exemplar source that minimizes SSD
        pp = get_source(p)
        # 2c Copy image data from exemplar to the patch
        a = p - self.n2 
        b = p + self.n2
        c = pp - self.n2
        d = pp + self.n2
        self.target[a:b,a:b,:] = self.source[c:d,c:d,:]
        # 3 Update confidence values of pixel values
        self.conf[a:b,a:b,:] = self.conf[p]
    
# %% Main Function
if __name__ == "__main__":
    inputFile = "fountain_girl.png"
    outputFile = exemplar(inputFile)