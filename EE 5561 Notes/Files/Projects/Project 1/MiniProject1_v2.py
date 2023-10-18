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


# %% Importing packages and functions

import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

       
# %% Main Exemplar class
class Exemplar():
    def __init__(self, patchSize=9, targetType ="Rectangle Coords", sourceType = "All", view=False):
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.imageCopy = None
        self.view = view
        
        # Exemplar class properties
        self.n = patchSize
        self.tt = targetType
        self.ss = sourceType
        
        # Image, target and Source are CONSTANT from INPUT
        self.image = None
        self.target = None
        self.grad = None
        
        # Target and Probabilities are Changing
        self.evolve = None
        self.confTot = None
        self.targetBool = None
        
        # Misc values
        self.n2 = int(0.5*(self.n-1))
        self.limit = 600
        self.count = 0
        
# %% Main Run Function     
    def run(self, iterations=None):
        if self.image is None or self.target is None or self.evolve is None:
            print("Choose image, target and source first!")
        else:
            self.gradX = np.zeros(self.shape)
            self.gradY = np.zeros(self.shape)
            self.gradMag = np.zeros(self.shape[:-1])
            for i in range(3):
                a, b = cv2.spatialGradient(self.image[:,:,i])
                self.gradX[:,:,i], self.gradY[:,:,i] = a, b
                self.gradMag += (a**2 + b**2)

            while self.count < self.limit and np.min(self.targetBool) == 0:
                self.update()
                if iterations is not None and self.count >= iterations:
                    break
                
            fig, ax = plt.subplots(figsize=(6,6), dpi=150)
            ax.imshow(self.evolve0, cmap="gray")
            ax.set_title("This is before inpainting")
            fig.tight_layout()
            ax.axis('off')
                        
            fig, ax = plt.subplots(figsize=(6,6), dpi=150)
            ax.imshow(self.evolve, cmap="gray")
            ax.set_title("This is the final inpainted photo")
            fig.tight_layout()
            ax.axis('off')

# %% Main Update Function
    def update(self): # Repeat until done:
        self.count += 1 
        print(f"Iteration: {self.count}")
        
        # 1a Identify the fill front
        ff = self.fillFront()
        
        # 1b Compute priorities of all P in the contour of target
        """ This Method is wrong and has to be changed """
        confs = self.compConf(ff)
        datas = self.compData(ff)
        
        # 2a Find patch with maximum priority
        priorities = confs * datas
        #breakpoint()
        bestPs = np.where(priorities == np.max(priorities))[0]
        chosenP = np.random.randint(0, len(bestPs))
        patchP = ff[bestPs[chosenP]]
        
        
        # 2b Find the exemplar source that minimizes SSD
        patchQ = self.getPatchQ(patchP)
        
        # 2c Copy image data from exemplar to the patch
        self.patchUp(patchP, patchQ)
        
        # 3 Update confidence values of pixel values
        self.updateVals(patchP, patchQ, confs[chosenP])
        
        fig, ax = plt.subplots(figsize=(6,6), dpi=150)
        ax.imshow(self.evolve, cmap="gray")
        ax.set_title(f"Iteration: {self.count}")
        fig.tight_layout()
        ax.axis('off')
        

# %% Get Inputs
    def inputImage(self, imageFile):
        image = cv2.imread(imageFile)   
        print(f"The input image has dimensions: {image.shape}")
        self.image = image
        self.shape = image.shape
        self.count = 1
    
    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.imageCopy, (self.start_x, self.start_y), (x, y), (0, 0, 0), 5)
                cv2.line(self.target, (self.start_x, self.start_y), (x, y), (0, 0, 0), 5)
                self.start_x, self.start_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
        
    def chooseTarget(self, coords=None):
        if self.image is not None:
            if self.tt == "Rectangle Coords":
                if coords == None:
                    print("Enter Rectangular Coordinates Please")
                else:
                    x1, x2, y1, y2 = coords
                    target = np.ones((self.shape[0], self.shape[1]), np.uint8)
                    target[x1:x2, y1:y2] = 0
                    self.target = target
            elif self.tt == "Free Draw":
                self.target = np.ones((self.shape[0], self.shape[1]), dtype=np.uint8)*255
                self.imageCopy = np.copy(self.image)
                
                
                cv2.namedWindow('Image')
                cv2.setMouseCallback('Image', self.draw_circle)
                while True:
                    cv2.imshow('Image', self.imageCopy)
                    # Press 'c' to clear the mask
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        self.imageCopy = np.copy(self.image)
                        self.target = np.ones((self.shape[0], self.shape[1]), dtype=np.uint8)*255
                    # Press 'q' to quit
                    elif key == ord('q'):
                        break
                cv2.destroyAllWindows()
                _, self.target = cv2.threshold(self.target, 128, 255, cv2.THRESH_BINARY)
                self.target = (self.target / 255).astype(np.uint8)
        else:
            print("Get an input first!")
            return False
        
    
        # changing
        self.confTot = np.copy(self.target).astype(float)
        self.targetBool = np.copy(self.target)
        # misc
        self.count = 1
        return True
    def chooseSource(self):
        if self.image is None and self.target is None:
            print("Choose an image and target first!")
            return False     
        else:
            if self.ss == "Padding":
                pass
            elif self.ss == "Rectangle":
                pass
            elif self.ss == "All":
                self.evolve = np.zeros(self.shape, dtype=np.uint8)
                for row in range(self.shape[0]):
                    for col in range(self.shape[1]):
                        self.evolve[row,col,:] = self.image[row,col,:] * self.target[row,col]
                """
                self.evolvesad = np.full(self.shape, np.nan)
                for row in range(self.shape[0]):
                    for col in range(self.shape[1]):
                        if self.target[row, col] == 1:
                            self.evolvesad[row,col,:] = self.image[row,col,:] 
                """
                self.evolve0 = np.copy(self.evolve)
            elif self.ss == "Free Draw":
                pass   
        
        self.allSource = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1]) if 
                     np.any(self.targetBool[i-self.n2:i+self.n2+1,j-self.n2:j+self.n2+1]==0) == False
                     and i-self.n2 > 0 and i+self.n2 < self.shape[0] 
                     and j-self.n2 > 0 and j+self.n2 < self.shape[1] ]
        
        self.compareWith = []
        for coords in range(len(self.allSource)):
            c, d = self.allSource[coords]
            self.compareWith.append( cv2.cvtColor(self.evolve[c-self.n2:c+self.n2+1,d-self.n2:d+self.n2+1], cv2.COLOR_BGR2LAB) )
        
        self.compareWith = np.asarray(self.compareWith)
            
        self.count = 0 
        return True
    
# %% Viewing Image, Target and Source 

    # Show the input image
    def showImage(self):
        if self.image is None:
            print("Empty Image")
            return False
        else:
            print(f"Image array has datatype: {self.image.dtype}")
            print(f"Image has size: {self.image.shape}")
            """
            fig, ax = plt.subplots(figsize=(6,6), dpi=150)
            ax.set_title("Original Image")
            ax.imshow(self.image, cmap="gray")
            fig.tight_layout()
            ax.axis('off')
            """
            cv2.namedWindow('Image')
            while True:
                cv2.imshow('Image', self.image)
                key = cv2.waitKey(1) & 0xFF
                # Press 'q' to quit
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
            return True
        
    # Show the target of the image   
    def showTarget(self):
        if self.image is None or self.target is None:
            print("Empty Image or Target")
            return False
        else:
            print(f"Target array has datatype: {self.target.dtype}")
            print(f"Target has size: {self.target.shape}")
    
            cv2.namedWindow('Image')
            while True:
                cv2.imshow('Image', self.target.astype(float))
                key = cv2.waitKey(1) & 0xFF
                # Press 'q' to quit
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
            return True
        
    # Show the source of the image
    def showSource(self):
        if self.image is None or self.target is None or self.evolve is None:
            print("Empty Image")
            return False
        else:
            print(f"Image array has datatype: {self.evolve.dtype}")
           
            cv2.namedWindow('Image')
            while True:
                cv2.imshow('Image', self.evolve)
                key = cv2.waitKey(1) & 0xFF
                # Press 'q' to quit
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
            return True

# %% Internal Functions
    def fillFront(self):
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        kernel = kernelx + kernely
        #breakpoint()
        heyhey = (np.abs(convolve2d(self.targetBool, kernel, mode='same', boundary='wrap'))*255).astype(int)
        
        coordinates = []
        for row in range(self.image.shape[0]):
            for col in range(self.image.shape[1]):
                if heyhey[row,col] != 0:
                    coordinates.append([row,col])
                    
        if self.view == True:
            fig, ax = plt.subplots()
            ax.imshow(heyhey, cmap="gray")
            ax.set_title("cmap")
        
        return coordinates
    
    
    def compConf(self, ff):
        confs = np.zeros(len(ff))
        if self.view is True:
            fig, ax = plt.subplots()
        
        for coords in range(len(confs)):
            a, b = ff[coords]
            confs[coords] = np.sum(self.confTot[a-self.n2:a+self.n2+1,b-self.n2:b+self.n2+1]) / (self.n**2)
            if self.view is True:
                ax.scatter(b, a, s=confs[coords])
                
        if self.view is True:
            print(f"min conf: {np.min(confs)}")
            print(f"max conf: {np.max(confs)}")
            g = np.argwhere(confs == np.max(confs))
            print(g)
            #for i in range(len(g)):
            #    print(f"{ff[g[i]]}")    
            print(ff[40])
            print(ff[275])
        
        return confs
    
    
    def compData(self, ff):
        ff = np.array(np.copy(ff))
        datas = np.zeros(len(ff))
        
        if self.view is True:
            fig, ax = plt.subplots()
        
        for coords in range(len(datas)):
            a, b = ff[coords]
            arr = self.gradMag[a-self.n2:a+self.n2+1,b-self.n2:b+self.n2+1]
            #breakpoint()
            i, j = np.unravel_index(np.argmax(arr), (self.n, self.n))
            #Fit a line to the contour points
            #breakpoint()
            [vx, vy, x, y] = cv2.fitLine(ff, cv2.DIST_L2, 0, 0.01, 0.01)
            t = np.arctan(vy / vx)
            normal = np.array([-np.sin(t), np.cos(t)])
            ii = a - self.n2
            jj = b - self.n2
            arrarr = [np.max(self.gradX[ii,jj]), np.max(self.gradY[ii,jj])]
            datas[coords] = np.abs(np.dot(arrarr, normal)) / 255
            if self.view is True:
                ax.scatter(b, a, s=datas[coords])
                
        if self.view is True: 
            print(f"min data: {np.min(datas)}")
            print(f"max data: {np.max(datas)}")
            g = np.argwhere(datas == np.max(datas))
            print(g)
            #breakpoint()
            
        return datas
        
    
    def getPatchQ(self, patchP):
        a, b = patchP
        dataOrig = self.evolve[a-self.n2:a+self.n2+1,b-self.n2:b+self.n2+1]
        dOM = self.targetBool[a-self.n2:a+self.n2+1,b-self.n2:b+self.n2+1]
        #fig, ax = plt.subplots()
        #ax.imshow(dOM)
        #breakpoint()
        DOM = np.stack([dOM] * 3, axis=2)
        #print(dOM)
        
        """np.
        for item in range(self.n**2):
            if nan_mask[item] == 0:
                compare[item] = cv2.cvtColor(data[item], cv2.COLOR_BGR2LAB)
        """
        
        compare = cv2.cvtColor(dataOrig, cv2.COLOR_BGR2LAB)
        
        ssd = np.zeros(len(self.allSource))
        
        for coords in range(len(self.allSource)):   
            ssd[coords] = np.sum(((self.compareWith[coords]-compare)*DOM)**2)
        #breakpoint()  
        if self.view is True:
            print(np.min(ssd[dOM]))
        saddest = np.where(ssd == np.min(ssd))[0]
        bb = np.random.randint(0, len(saddest))
        patchQ = self.allSource[saddest[bb]]
        
        if self.view is True:
            fig, ax = plt.subplots()
            ax.imshow(dataOrig)
            
            fig, ax = plt.subplots()
            ax.imshow(compare)
            
            fig, ax = plt.subplots()
            ax.imshow(self.compareWith[bb])
        
        return patchQ
    
    def patchUp(self, patchP, patchQ):
        a = patchP[0]
        b = patchP[1]
        c = patchQ[0]
        d = patchQ[1]
        #print(patchQ)
        self.evolve[a-self.n2:a+self.n2+1,
                    b-self.n2:b+self.n2+1] = self.evolve[c-self.n2:c+self.n2+1
                                                        ,d-self.n2:d+self.n2+1]
        return True
        
    def updateVals(self, patchP, patchQ, maxConf):
        a = patchP[0]
        b = patchP[1]
        
        # Update Probabilities
        self.confTot[a-self.n2:a+self.n2+1,
                     b-self.n2:b+self.n2+1] = maxConf
        
        # Update targetBool
        self.targetBool[a-self.n2:a+self.n2+1,
                        b-self.n2:b+self.n2+1] = 1
        return True 

