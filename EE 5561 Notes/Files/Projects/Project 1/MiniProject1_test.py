
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import scipy

image = img.imread("images/fountain_girl.png") # importing using cv2 switches the color channels
#plt.imshow(image)
image = image[:,:,:-1]
# blue green red
mask = np.zeros(image.shape[:-1])
# x is the rows from up
# y is the columns from right
mask[800:,250:750] = 1
nmask = np.abs(1-mask)
#plt.imshow(mask, cmap="gray")

#target = np.ones(image.shape)
source = np.ones(image.shape)
for i in range(3):
    #target[:,:,i] = image[:,:,i] * mask
    source[:,:,i] = image[:,:,i] * mask
#plt.imshow(source)

probs = np.copy(mask)
print(probs.shape)
ff = scipy.ndimage.sobel(probs, 0)**2 + scipy.ndimage.sobel(probs, 1)**2
#plt.imshow(ff)
contours = np.where(ff > 9, 1, 0)
#plt.imshow(contours)
size = 9
myFilter = 1/size**2 * np.ones((size,size))
newProbs = scipy.signal.convolve2d(probs, myFilter, mode='same', boundary='wrap') * mask
plt.imshow(newProbs)
print(newProbs.shape)
ab = np.unravel_index(np.argmax(newProbs), image.shape[:-1])
print(ab)
fig, ax = plt.subplots()
ax.set_ylim([0,image.shape[0]])
ax.set_xlim([0,image.shape[1]])
ax.plot(ab[0], ab[1], markersize=100)

xgrad = scipy.ndimage.sobel(image, 0)
ygrad = scipy.ndimage.sobel(image, 1)
xsad = xgrad[:,:,0]**2 + xgrad[:,:,1]**2 + xgrad[:,:,2]**2
Xsad = xsad**2
ysad = ygrad[:,:,0]**2 + ygrad[:,:,1]**2 + ygrad[:,:,2]**2
Ysad = ysad**2
mag_grad = np.sqrt(Xsad+Ysad)
newProbs = scipy.signal.convolve2d(mag_grad, myFilter, mode='same', boundary='wrap') * mask

"""
Image Creation
"""
def createImages():
    bw = np.zeros((200,200))
    bw[:128,:] = 100
    bw[128:,:] = 200
    cv2.imwrite("images/blackWhite.png", bw)
    
def threshold(image, limit, saveName = None):
    imageThresh = np.ones(image.shape[:-1],dtype=np.uint8)
    imageThresh[image[:,:,3] < limit] = 0
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    ax.imshow(imageThresh, cmap="gray")
    if saveName is not None:
        imageThresh = np.copy(imageThresh).T
        cv2.imwrite(saveName, imageThresh)
        print("Image saved!")
    return imageThresh

def inputImage(filename):
    image = np.asarray(img.imread(filename))
    print(filename.dtype)
    print(filename.shape)
    
# To save to a png file without converting it auto into a 4 channel use this   
#cv2.imwrite('images/post Process/triangle2.png', imageThresh) 
