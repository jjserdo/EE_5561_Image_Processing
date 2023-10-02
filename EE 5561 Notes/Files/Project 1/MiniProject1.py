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
import cv2
       
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
        self.gradmax = None
        self.lab = None
        self.n2 = int(0.5*(self.n-1))
        self.limit = 600
        self.count = 1
        
    def inputImage(self, imageFile):
        image = cv2.imread(imageFile)   
        print(f"The input image has dimensions: {image.shape}")
        self.image = image
        self.shape = image.shape
        self.count = 1
        
    def showInput(self):
        if self.image is not None:
            print("Empty Image")
            return False
        else:
            while (True):
                cv2.imshow("Display: Press Esc to exit", self.image)
                if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
                    break
            return True
    
    def chooseTargetClick(self):
        def inspect(event,y,x):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                print(f"image[{y}, {x}]")
                cv2.destroyAllWindows()
                return y, x
                    
        cv2.namedWindow("Target", self.image)
        cv2.setMouseCallback("Target", self.image)
        
        while(True):
            cv2.imshow('Target', self.image)
            if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
                break
                cv2.destroyAllWindows()
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
    
            elif self.tt == "Rectangle Click":
                print("Click for upper left vertex of rectangle")
                y1, x1 = self.inspect()
                print("Click for lower right vertex of rectangle")
                y2, x2 = self.inspect()
                target = np.ones(self.image)
                target[x1:x2, y1:y2, :] = 0

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
        
        self.showTarget()
        
        return True
        
    def showTarget(self):
        if self.image is not None:
            print("Empty Target")
            return False
        else:
            while (True):
                cv2.imshow("Mask", self.mask)
                if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
                    break
            return True
               
    def chooseSource(self):
        if self.image is not None and self.mask is not None:
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
        
        while (True):
            cv2.imshow("Source", self.mask)
            if cv2.waitKey(20) & 0xFF == 27: # ASCII character = 27 which is 'escape'
                break
        
        return True
        
    ### Run Funntion
    def run(self):
        if self.image is not None and self.mask is not None and self.source is not None:
            a, b = cv2.spatialGradient(self.image)
            self.grad = np.vstack(a, b).T
            while self.count < self.limit and np.min(self.conf) != 0:
                self.update()
        else:
            print("Choose image, target and source first!")
    
    """
    def comp_prio(self,listofP):
        g = listofP.shape[0]
        confidence = np.zeros(g)
        for i in range(g):
            px = listofP[g,0]
            py = listofP[g,1]
            confidence[i] = np.sum(self.conf[px-self.n2:px+self.n2,py-self.n2:py+self.n2])
        return confidence
    """
    
    def comp_DATA(self):
        h = self.grad[0]**2+self.grad[1]**2
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                a = i - self.n2
                b = i - self.n2
                c = j - self.n2
                d = j - self.n2
                hh = np.unravel(np.argmax(h[a:b,c:d]), h.shape)
                self.gradmax[0,i,j] = self.grad[0,h[0],h[1]]
                self.gradmax[1,i,j] = self.grad[1,h[0],h[1]]
    
    def get_sourcePatch(self,p):
        compare = cv2.cvtColor(self.evolve[p[0]-self.n2:p[0]+self.n2,p[1]-self.n2:p[1]+self.n2,:], cv2.COLOR_BGR2LAB)
        g = 1
        ssd = np.zeros(1)
        for i in range(g):
            sourceX = a
            sourceY = b
            ssd[i] = np.sum((self.lab[a-self.n2:a+self.n2,b-self.n2:b+self.n2,:]-compare)**2)
        source_patch = np.unravel_index(np.argmax(ssd))
        return source_patch
        
    def update(self): # Repeat until done:
        self.count += 1
        # 1a Identify the fill front
        ff = cv2.Canny(self.evolve, 0, 0)
        # 1b Compute priorities of all P in the contour of target
        confWhole = np.zeros(ff.shape)
        dataWhole = np.zeros(ff.shape)
        for i in range(ff.shape[0]):
            for j in range(ff.shape[1]):
                confWhole[i,j] = np.sum(self.conf[px-self.n2:px+self.n2,py-self.n2:py+self.n2])
                dataWhole[i,j] = np.unravel(np.argmax(self.grad[0]**2+self.grad[1]**2), self.grad.shape[1:]
        priorities = confWhole * dataWhole
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
    pass



# adder = np.ones(self.n, self.n)
# priorities = cv2.convolve(probs, adder)
