import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import imageio as iio
from scipy.fftpack import dct
from scipy.fftpack import idct

def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm = 'ortho'))

def ifft2c(freq):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(freq), norm = 'ortho'))

def dct2c(img):
    return dct(dct(img, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

def idct2c(img_dct):
    return idct(idct(img_dct, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

def moving_average_filter(img, kernel_size):
    kernel = np.ones((kernel_size[0], kernel_size[1])) / (np.product(kernel_size))
    return convolve2d(img, kernel, mode='same', boundary='wrap')

# %%  Load the images as numpy complex arrays
lena_img = np.complex64(plt.imread('lena512.bmp'))
man_img  = np.complex64(iio.imread('man.png'))
coin_img = np.complex64(plt.imread('eight.tif'))

# %% #############################################  4)a.  ####################################################################
plt.figure(figsize=(9,10))
plt.subplot(3,3,1)
plt.imshow(np.log(np.abs(fft2c(lena_img))), cmap='gray')
plt.axis('off'); plt.title('FFT Magnitude')
plt.subplot(3,3,2)
plt.imshow(np.angle(fft2c(lena_img)), cmap='gray')
plt.axis('off'); plt.title('FFT Phase')
plt.subplot(3,3,3)
plt.imshow(np.abs(lena_img), cmap='gray')
plt.axis('off'); plt.title('Lena')

plt.subplot(3,3,4)
plt.imshow(np.log(np.abs(fft2c(man_img))), cmap='gray')
plt.axis('off'); plt.title('FFT Magnitude')
plt.subplot(3,3,5)
plt.imshow(np.angle(fft2c(man_img)), cmap='gray')
plt.axis('off'); plt.title('FFT Phase')
plt.subplot(3,3,6)
plt.imshow(np.abs(man_img), cmap='gray')
plt.axis('off'); plt.title('Man')

new_FFT = np.abs(fft2c(man_img)) * (fft2c(lena_img)/np.abs(fft2c(lena_img)))

plt.subplot(3,3,7)
plt.imshow(np.log(np.abs(fft2c(man_img))), cmap='gray')
plt.axis('off'); plt.title('FFT Magnitude')
plt.subplot(3,3,8)
plt.imshow(np.angle(fft2c(lena_img)), cmap='gray')
plt.axis('off'); plt.title('FFT Phase')
plt.subplot(3,3,9)
plt.imshow(np.abs(ifft2c(new_FFT)), cmap='gray')
plt.axis('off'); plt.title('New Image')
plt.tight_layout()

# %% #############################################  4)b.  ####################################################################
[Nx,Ny] = lena_img.shape
# DCT Thresholding
DCT_lena_thr = np.zeros_like(lena_img)
for x in range(0,Nx,8):
    for y in range(0,Ny,8):
        dct_patch = dct2c(lena_img[x:x+8,y:y+8])
        DCT_lena_thr[x:x+8,y:y+8] = idct2c(dct_patch * (np.abs(dct_patch)>np.sort(np.abs(dct_patch.flatten()))[-10]))
# DFT Thresholding
DFT_lena_thr = np.zeros_like(lena_img)
for x in range(0,Nx,8):
    for y in range(0,Ny,8):
        dft_patch = fft2c(lena_img[x:x+8,y:y+8])
        DFT_lena_thr[x:x+8,y:y+8] = ifft2c(dft_patch * (np.abs(dft_patch)>np.sort(np.abs(dft_patch.flatten()))[-10]))

plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
plt.imshow(np.abs(DCT_lena_thr), cmap='gray')
plt.axis('off'); plt.title('After DCT')
plt.subplot(1,2,2)
plt.imshow(np.abs(DFT_lena_thr), cmap='gray')
plt.axis('off'); plt.title('After DFT')
plt.tight_layout()

# %% #############################################  4)c.  ####################################################################
lowpass_coin = moving_average_filter(coin_img, [3,3])
highpass_coin = coin_img - lowpass_coin
sharpened_coin = coin_img + 2*(highpass_coin)

plt.figure(figsize=(9,8))
plt.subplot(2,2,1)
plt.imshow(np.abs(coin_img), cmap='gray', vmin=0, vmax=255)
plt.axis('off'); plt.title('Coins')
plt.subplot(2,2,2)
plt.imshow(np.abs(lowpass_coin), cmap='gray', vmin=0, vmax=255)
plt.axis('off'); plt.title('Low Pass')
plt.subplot(2,2,3)
plt.imshow(np.abs(highpass_coin), cmap='gray', vmin=0, vmax=255)
plt.axis('off'); plt.title('High Pass')
plt.subplot(2,2,4)
plt.imshow(np.abs(sharpened_coin), cmap='gray', vmin=0, vmax=255)
plt.axis('off'); plt.title('After Sharpening')
plt.tight_layout()



