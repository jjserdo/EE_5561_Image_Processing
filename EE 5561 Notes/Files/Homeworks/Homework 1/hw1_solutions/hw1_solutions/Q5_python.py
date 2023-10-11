import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def get_psnr(x_orig, x):
    nominator = np.max(abs(x_orig)**2)
    denominator = np.mean(abs(x - x_orig)**2)
    return 10*np.log10(nominator/denominator)

def moving_average_filter(img, kernel_size):
    kernel = np.ones((kernel_size[0], kernel_size[1])) / (np.product(kernel_size))
    return convolve2d(img, kernel, mode='same', boundary='wrap')

def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm = 'ortho'))

def ifft2c(freq):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(freq), norm = 'ortho'))

def LPF(image, filter_size):
    lowpassfilter = np.zeros_like(image, dtype=bool)
    lowpassfilter[image.shape[0]//2-filter_size[0]//2:image.shape[0]//2+filter_size[0]//2,image.shape[1]//2-filter_size[1]//2:image.shape[1]//2+filter_size[1]//2] = True
    return ifft2c(fft2c(image)*lowpassfilter)

def HPF(image, filter_size):
    lowpassfilter = np.ones_like(image, dtype=bool)
    lowpassfilter[image.shape[0]//2-filter_size[0]//2:image.shape[0]//2+filter_size[0]//2,image.shape[1]//2-filter_size[1]//2:image.shape[1]//2+filter_size[1]//2] = False
    return ifft2c(fft2c(image)*lowpassfilter)

# Load the bmp image as numpy ndarray
lena_img = np.complex64(plt.imread('lena512.bmp'))

# %% #############################################  5)a.  ####################################################################
# Take the Fourier Transform
freq_domain = fft2c(lena_img)
phase_spec = np.angle(freq_domain)
plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.abs(lena_img), cmap='gray')
plt.title('Lena')
plt.subplot(1,3,2)
plt.imshow(np.log(np.abs(freq_domain)), cmap='gray')
plt.title('FFT Magnitude')
plt.subplot(1,3,3)
plt.imshow(np.angle(freq_domain), cmap='gray')
plt.title('FFT Phase')
plt.tight_layout()
    
# %% ##############################################  5)b.  ####################################################################
img_lpf128 = LPF(lena_img, filter_size=[128,128])
psnr1 = get_psnr(lena_img,img_lpf128)
img_lpf32  = LPF(lena_img, filter_size=[32 ,32 ])
psnr2  = get_psnr(lena_img,img_lpf32)
plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.abs(lena_img), cmap='gray')
plt.title('Original image')
plt.subplot(1,3,2)
plt.imshow(np.abs(img_lpf128), cmap='gray')
plt.title('Low Pass Filter [128x128]')
plt.text(7, 500, f'PSNR: {psnr1:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.subplot(1,3,3)
plt.imshow(np.abs(img_lpf32), cmap='gray')
plt.title('Low Pass Filter [32x32]')
plt.text(7, 500, f'PSNR: {psnr2:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.tight_layout()

# %% ##############################################  5)c.  ####################################################################
img_hpf128 = HPF(lena_img, filter_size=[128,128])
psnr3 = get_psnr(lena_img,img_hpf128)
img_hpf32  = HPF(lena_img, filter_size=[32 ,32 ])
psnr4  = get_psnr(lena_img,img_hpf32)
plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.abs(lena_img), cmap='gray')
plt.title('Original image')
plt.subplot(1,3,2)
plt.imshow(np.abs(img_hpf128), cmap='gray')
plt.title('High Pass Filter [128x128]')
plt.text(7, 500, f'PSNR: {psnr3:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.subplot(1,3,3)
plt.imshow(np.abs(img_hpf32), cmap='gray')
plt.title('High Pass Filter [32x32]')
plt.text(7, 500, f'PSNR: {psnr4:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.tight_layout()

# %% ##############################################  5)d.  ####################################################################
freq_img_384 = np.copy(freq_domain)
freq_img_384[0:512-384] += freq_domain[384-512::]
freq_img_384[384-512::] += freq_domain[0:512-384]
freq_img_384[:,0:512-384] += freq_domain[:,384-512::]
freq_img_384[:,384-512:] += freq_domain[:,0:512-384]
sampled_img_384 = ifft2c(freq_img_384)

freq_img_256 = np.copy(freq_domain)
freq_img_256[0:512-256] += freq_domain[256-512::]
freq_img_256[256-512::] += freq_domain[0:512-256]
freq_img_256[:,0:512-256] += freq_domain[:,256-512::]
freq_img_256[:,256-512:] += freq_domain[:,0:512-256]
sampled_img_256 = ifft2c(freq_img_256)

plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.log(np.abs(freq_domain)), cmap='gray')
plt.title('Original kspace')
plt.subplot(1,3,2)
plt.imshow(np.log(np.abs(freq_img_384)), cmap='gray')
plt.title('1/384mm sampling')
plt.subplot(1,3,3)
plt.imshow(np.log(np.abs(freq_img_256)), cmap='gray')
plt.title('1/256mm sampling')
plt.tight_layout()

plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.abs(lena_img), cmap='gray')
plt.title('Original image')
plt.subplot(1,3,2)
plt.imshow(np.abs(sampled_img_384), cmap='gray')
plt.title('1/384mm sampling')
plt.subplot(1,3,3)
plt.imshow(np.abs(sampled_img_256), cmap='gray')
plt.title('1/256mm sampling')
plt.tight_layout()

# %% ##############################################  5)e.  ####################################################################    
# Convolve image with 3x3 and 7x7 kernels
convolved_img_3x3 = moving_average_filter(lena_img, [3,3])
psnr5 = get_psnr(lena_img, convolved_img_3x3)
convolved_img_7x7 = moving_average_filter(lena_img, [7,7])
psnr6 = get_psnr(lena_img, convolved_img_7x7)
plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.abs(lena_img), cmap='gray')
plt.title('Original image')
plt.subplot(1,3,2)
plt.imshow(np.abs(convolved_img_3x3), cmap='gray')
plt.title('3x3 Moving Average Filter')
plt.text(7, 500, f'PSNR: {psnr5:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.subplot(1,3,3)
plt.imshow(np.abs(convolved_img_7x7), cmap='gray')
plt.title('7x7 Moving Average Filter')
plt.text(7, 500, f'PSNR: {psnr6:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.tight_layout()
# Create a noisy image where sigma=10 and mean=0
noise = np.random.normal(0, 10, lena_img.shape)
noisy_img = lena_img + noise
# Convolve image with 3x3 and 7x7 kernels
convolved_img_3x3 = moving_average_filter(noisy_img, [3,3])
psnr7 = get_psnr(lena_img, convolved_img_3x3)
convolved_img_7x7 = moving_average_filter(noisy_img, [7,7])
psnr8 = get_psnr(lena_img, convolved_img_7x7)
plt.figure(figsize=(9,3.5))
plt.subplot(1,3,1)
plt.imshow(np.abs(noisy_img), cmap='gray')
plt.title('Noisy image')
plt.subplot(1,3,2)
plt.imshow(np.abs(convolved_img_3x3), cmap='gray')
plt.title('3x3 Moving Average Filter')
plt.text(7, 500, f'PSNR: {psnr7:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.subplot(1,3,3)
plt.imshow(np.abs(convolved_img_7x7), cmap='gray')
plt.title('7x7 Moving Average Filter')
plt.text(7, 500, f'PSNR: {psnr8:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
plt.tight_layout()