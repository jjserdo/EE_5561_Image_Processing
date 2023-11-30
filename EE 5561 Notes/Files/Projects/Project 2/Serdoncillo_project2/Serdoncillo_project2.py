'''
Justine Serdoncillo
EE 5561 - Image Processing
Mini-Project 2
November 9, 2023
'''

"""
Implementation of 
"A non-local algorithm for image denoising"
A. Buades, B. Coll, J.M. Morel
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
from tkinter import *
import imageio as iio
from NonLocalDenoise import NonLocal
from PersonalDenoise import jjGausian, jjMedian, jjPGD, jjADMM
from PersonalNoise import jjGaussNoise, jjSnP

# %% Metrics Functions
def get_psnr(x_orig, x):
    nominator = np.max(abs(x_orig)**2)
    denominator = np.mean(abs(x - x_orig)**2)
    psnr = 10*np.log10(nominator/denominator)
    return psnr
def get_mse(x_orig, x):
    mse = np.mean((x_orig - x) ** 2)
    return mse


# %% Main Function
if __name__ == "__main__":
    images = []
    lena_img = np.complex64(iio.imread('images/lena5120.png'))
    man_img  = np.complex64(iio.imread('images/man0.png'))
    coin_img = np.complex64(iio.imread('images/eight0.png'))
    cman_img = np.complex64(iio.imread('images/cameraman0.png'))
    images.append(lena_img)
    images.append(man_img)
    images.append(coin_img)
    images.append(cman_img)
    
    """
    for ii in range(len(images)):
        fig, ax = plt.subplots()
        ax.imshow(np.abs(images[ii]).astype(np.uint8), cmap='gray')
    """
    
    
    gnoise = []
    snoise = []
    jjGaussNoise = jjGaussNoise(mean=0.0, var=160)
    jjSnP = jjSnP(salt_prob=0.01, pepper_prob=0.01)
    for ii in range(len(images)):
        gn  = jjGaussNoise.noise(images[ii].astype(np.int16))
        gnoise.append(gn)
        cv2.imwrite("images/noised/g" +str(ii)+ ".png", gn)
        
        snp = jjSnP.noise(images[ii].astype(np.int16))
        snoise.append(snp)
        cv2.imwrite("images/noised/s" +str(ii)+ ".png", snp)    
    
    
    jj1 = jjGausian()  
    jj2 = jjMedian()
    jj3 = jjPGD()
    jj4 = jjADMM()
    jj5 = NonLocal()
    gdenoised = []
    sdenoised = []
    for ii in range(len(images)):
        #gf = jj1.denoise(gnoise[ii])
        #mf = jj2.denoise(gnoise[ii])
        #tvd1 = jj3.denoise(gnoise[ii]) 
        #tvd2 = jj4.denoise(gnoise[ii])
        nl = jj5.denoise(gnoise[ii])
        #cv2.imwrite("images/denoised/ggf" +str(ii)+ ".png", gf)
        #cv2.imwrite("images/denoised/gmf" +str(ii)+ ".png", mf)
        #cv2.imwrite("images/denoised/gtvd1" +str(ii)+ ".png", tvd1)
        #cv2.imwrite("images/denoised/gtvd2" +str(ii)+ ".png", tvd2)
        cv2.imwrite("images/denoised/gnl" +str(ii)+ ".png", nl)
        #gdenoised.append(gf)
        #gdenoised.append(mf)
        #gdenoised.append(tvd1)
        #gdenoised.append(tvd2)
        gdenoised.append(nl)
        
        #gf = jj1.denoise(snoise[ii])
        #mf = jj2.denoise(snoise[ii])
        #tvd1 = jj3.denoise(snoise[ii]) 
        #tvd2 = jj4.denoise(snoise[ii])
        nl = jj5.denoise(snoise[ii])
        #cv2.imwrite("images/denoised/sgf" +str(ii)+ ".png", gf)
        #cv2.imwrite("images/denoised/smf" +str(ii)+ ".png", mf)
        #cv2.imwrite("images/denoised/stvd1" +str(ii)+ ".png", tvd1)
        #cv2.imwrite("images/denoised/stvd2" +str(ii)+ ".png", tvd2)
        cv2.imwrite("images/denoised/snl" +str(ii)+ ".png", nl)
        #sdenoised.append(gf)
        #sdenoised.append(mf)
        #sdenoised.append(tvd1)
        #sdenoised.append(tvd2)
        sdenoised.append(nl)
    for ii in range(len(images)):
        print(f"PSNR value for gaussian image {ii}: {get_psnr(images[ii], gnoise[ii])}" )
    for ii in range(len(images)):
        print(f"PSNR value for gaussian image {ii}: {get_psnr(images[ii], gdenoised[ii])}" )
    for ii in range(len(images)):
        print(f"MSE value for gaussian image {ii}: {get_mse(images[ii], gnoise[ii])}" )
    for ii in range(len(images)):
        print(f"MSE value for gaussian image {ii}: {get_mse(images[ii], gdenoised[ii])}" )
        
    for ii in range(len(images)):
        print(f"PSNR value for gaussian image {ii}: {get_psnr(images[ii], snoise[ii])}" )
    for ii in range(len(images)):
        print(f"PSNR value for gaussian image {ii}: {get_psnr(images[ii], sdenoised[ii])}" )
    for ii in range(len(images)):
        print(f"MSE value for gaussian image {ii}: {get_mse(images[ii], snoise[ii])}" )
    for ii in range(len(images)):
        print(f"MSE value for gaussian image {ii}: {get_mse(images[ii], sdenoised[ii])}" )
    """
    for ii in range(len(images)):
        for jj in range(5):
            print(f"PSNR value for gaussian image {ii}: {get_psnr(images[ii], gnoise[ii*5+jj])}")
    for ii in range(len(images)):
        for jj in range(5):
            print(f"PSNR value for SnP image {ii}: {get_psnr(images[ii], snoise[ii*5+jj])}")
    for ii in range(len(images)):
        for jj in range(5):
            print(f"MSE value for gaussian image {ii}: {get_mse(images[ii], gnoise[ii*5+jj])}")
    for ii in range(len(images)):
        for jj in range(5):
            print(f"MSE value for SnP image {ii}: {get_mse(images[ii], snoise[ii*5+jj])}")
            
    for ii in range(len(images)):
        for jj in range(5):
            print(f"PSNR value for gaussian image {ii}: {get_psnr(images[ii*5+jj], gdenoised[ii*5+jj])}")
    for ii in range(len(images)):
        for jj in range(5):
            print(f"PSNR value for SnP image {ii}: {get_psnr(images[ii], sdenoised[ii*5+jj])}")
    for ii in range(len(images)):
        for jj in range(5):
            print(f"MSE value for gaussian image {ii}: {get_mse(images[ii], gdenoised[ii*5+jj])}")
    for ii in range(len(images)):
        for jj in range(5):
            print(f"MSE value for SnP image {ii}: {get_mse(images[ii], sdenoised[ii*5+jj])}")
    """
        
    nl1 = NonLocal(h=10, searchWindow=13, squareNeighborhood=7)
    nl2 = NonLocal(h=5, searchWindow=21, squareNeighborhood=7)
    nl3 = NonLocal(h=10, searchWindow=21, squareNeighborhood=3)
    nl4 = NonLocal(h=5, searchWindow=13, squareNeighborhood=3)
    nl5 = NonLocal(h=7, searchWindow=17, squareNeighborhood=5)
    o0 = jj5.denoise(gnoise[0])
    o1 = nl1.denoise(gnoise[0])
    o2 = nl1.denoise(gnoise[0])
    o3 = nl1.denoise(gnoise[0])
    o4 = nl1.denoise(gnoise[0])
    o5 = nl1.denoise(gnoise[0])
    cv2.imwrite("images/hyper/g0.png", o0)
    cv2.imwrite("images/hyper/g1.png", o1)
    cv2.imwrite("images/hyper/g2.png", o2)
    cv2.imwrite("images/hyper/g3.png", o3)
    cv2.imwrite("images/hyper/g4.png", o4)
    cv2.imwrite("images/hyper/g5.png", o5)
    print(f"PSNR value for gaussian lena image: {get_psnr(images[0], o0)}" )
    print(f"MSE value for gaussian lena image: {get_mse(images[0], o0)}")
    print(f"PSNR value for gaussian lena image: {get_psnr(images[0], o1)}" )
    print(f"MSE value for gaussian lena image: {get_mse(images[0], o1)}")
    print(f"PSNR value for gaussian lena image: {get_psnr(images[0], o2)}" )
    print(f"MSE value for gaussian lena image: {get_mse(images[0], o2)}")
    print(f"PSNR value for gaussian lena image: {get_psnr(images[0], o3)}" )
    print(f"MSE value for gaussian lena image: {get_mse(images[0], o3)}")
    print(f"PSNR value for gaussian lena image: {get_psnr(images[0], o4)}" )
    print(f"MSE value for gaussian lena image: {get_mse(images[0], o4)}")
    print(f"PSNR value for gaussian lena image: {get_psnr(images[0], o5)}" )
    print(f"MSE value for gaussian lena image: {get_mse(images[0], o5)}")
    
    
    print("DONE")
    
    

    
    
    
