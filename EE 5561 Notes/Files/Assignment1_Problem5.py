'''
Justine Serdoncillo
EE 5561 - Image Processing
Problem Set 1
September 22, 2023
'''

'''
np.fft.fft2, - 2d FFT
np.fft.fftshift, - shift
np.fft.ifft2, - invers 2D FFT
np.fft.ifftshift - inverse shift
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy

lena = img.imread('lena512.bmp')
print(lena.dtype)
print(lena.shape)
plt.imshow(lena)
plt.show()

lenaArr = asarray(lena)
print(type(lenaArr))
print(lenaArr.shape)

# Calculate the 2D FFT of the Lena image.
# Plot its logarithmic magnitude spectrum, and phase spectrum
p5a = np.fft.fft2(lena) 

# Apply low-pass filters of 128x128 and 32x32 
'''
https://stackoverflow.com/questions/66935821/how-to-apply-a-lpf-and-hpf-to-a-fft-fourier-transform
'''
# Calculate the peak signal-to-noise ratio (PSNR)
def PSNR(newImage, oldImage):
    s1, s2 = oldImage.shape
    top = np.max(np.abs(oldImage)**2)
    bottom = 1/(s1*s2)*np.sum(np.abs(newImage-oldImage)**2)
    return 10*np.log10(top/bottom)
# Apply high-pass filters and calculate the PSNRs

# Sample image with different sample spacings and report PSNRs
# 1/384 mm sample spacing
# 1/256 mm sample spacing

'''
scipy.signal.convolve2d
'''

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Problem 5 D')
ax1.plot(x,y)
ax2.plot(x,y)

# Implement moving average filter of 2x3 and 7x7 filters
#scipy.signal.convolve2d
# Add random Gaussian noise
