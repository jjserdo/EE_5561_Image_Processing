'''
Justine Serdoncillo
EE 5561 - Image Processing
Problem Set 1
September 22, 2023
'''

'''
np.fft.fft2, - 2d FFT
np.fft.fftshift, - shift
np.fft.ifft2, - inverse 2D FFT
np.fft.ifftshift - inverse shift
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy

def PSNR(orig, new):
    xx, yy = orig.shape
    top = np.amax(np.abs(orig)**2)
    allsum = 0
    for m in range(xx):
        for n in range(yy): 
            allsum += np.abs(new[m,n] - orig[m,n])**2
    bot = 1/(xx*yy) * allsum 
    return 10*np.log10(top/bot)

# %% Part A
def probA(file):
    image = img.imread(file)    
    lena = np.asarray(image)
    print(type(lena))
    print(lena.shape)
    xx, yy = lena.shape
    
    # Calculate the 2D FFT of the Lena image.
    # Plot its logarithmic magnitude spectrum, and phase spectrum
    ffLena = np.fft.fftshift(np.fft.fft2(lena))
    magLena = np.log(np.abs(ffLena))
    phaLena = np.angle(ffLena)
    
    fig, ax = plt.subplots(dpi=300) 
    ax.imshow(lena, cmap='gray')
    ax.set_axis_off()
    ax.set_title("Our very own Lena of EE 5561")
    fig.savefig('images/alena.png', bbox_inches='tight', pad_inches=0)
    
    fig, (mag, pha) = plt.subplots(1, 2, dpi=300)
    mag.imshow(magLena, cmap='gray')
    mag.set_axis_off()
    mag.set_title("Logarithmic Magnitude")
    pha.imshow(phaLena, cmap='gray')
    pha.set_axis_off()
    pha.set_title("Phase")
    fig.savefig('images/amagpha.png', bbox_inches='tight', pad_inches=0)
    

# %% Part B
def probB(file, sizes):
# Apply low-pass filters of 128x128 and 32x32 
# Calculate the peak signal-to-noise ratio (PSNR)
    image = img.imread(file)
    lena = np.asarray(image)
    xx, yy = lena.shape
    ffLena = np.fft.fftshift(np.fft.fft2(lena))
    iffLena = np.fft.ifft2(np.fft.ifftshift(ffLena)).real
    
    """
    fig, ax = plt.subplots(dpi=300) 
    ax.imshow(lena, cmap='gray')
    ax.set_axis_off()
    
    fig, ax = plt.subplots(dpi=300) 
    ax.imshow(iffLena, cmap='gray')
    ax.set_axis_off()
    """
    
    tt = len(sizes)
    
    lowLena = np.zeros((tt,xx,yy), np.complex_)
    highLena = np.zeros((tt,xx,yy), np.complex_)
    for i in range(tt):
        aa = int(round(0.5*(xx-sizes[i])))
        bb = int(round(0.5*(xx+sizes[i])))
        lowFilter = np.zeros((xx,yy), np.complex_)
        lowFilter[aa:bb,aa:bb] = 1
        lowLena[i] = lowFilter * ffLena
        highFilter = np.ones((xx,yy), np.complex_)
        highFilter[aa:bb,aa:bb] = 0
        highLena[i] = highFilter * ffLena
    
    fig, axs = plt.subplots(1,tt, dpi=300)
    fig1, axs1 = plt.subplots(1,tt, dpi=300)
    for i in range(tt):
        hi = np.log(np.abs(lowLena[i]))
        there = np.fft.ifft2(np.fft.ifftshift(lowLena[i])).real
        axs[i].imshow(hi, cmap='gray')
        axs1[i].imshow(there, cmap='gray')
        axs[i].set_axis_off()
        axs1[i].set_axis_off()
        axs[i].set_title(f'{sizes[i]}x{sizes[i]}')
        axs1[i].set_title(f'{sizes[i]}x{sizes[i]}')
        print(f'low pass PSNR for {sizes[i]}x{sizes[i]}: {PSNR(lena,there)}')
    fig.savefig('images/blow.png', bbox_inches='tight', pad_inches=0)
    fig1.savefig('images/blowlena.png', bbox_inches='tight', pad_inches=0)
    

# %% Part B
# Apply high-pass filters and calculate the PSNRs
    
    fig, axs = plt.subplots(1,tt, dpi=300)
    fig1, axs1 = plt.subplots(1,tt, dpi=300)
    for i in range(tt):
        hi = np.log(np.abs(highLena[i]))
        there = np.fft.ifft2(np.fft.ifftshift(highLena[i])).real
        axs[i].imshow(hi, cmap='gray')
        axs1[i].imshow(there, cmap='gray')
        axs[i].set_axis_off()
        axs1[i].set_axis_off()
        axs[i].set_title(f'{sizes[i]}x{sizes[i]}')
        axs1[i].set_title(f'{sizes[i]}x{sizes[i]}')
        print(f'high pass PSNR for {sizes[i]}x{sizes[i]}: {PSNR(lena,there)}')
    fig.savefig('images/bhigh.png', bbox_inches='tight', pad_inches=0)
    fig1.savefig('images/bhighlena.png', bbox_inches='tight', pad_inches=0)


# %% Part C
def probC(file, sizes):
# Sample image with different sample spacings and report PSNRs
# 1/384 mm sample spacing
# 1/256 mm sample spacing in both dimensions
    image = img.imread(file)
    lena = np.asarray(image)
    xx, yy = lena.shape
    aa = int(xx/2)
    tt = len(sizes)
    ffLena = np.fft.fftshift(np.fft.fft2(lena))
    
    fig, axs = plt.subplots(1,tt+1, dpi=300)
    axs[0].imshow(lena, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('Original')
    
    fig1, axs1 = plt.subplots(1,tt+1, dpi=300)
    axs1[0].imshow(np.log(np.abs(ffLena)), cmap='gray')
    axs1[0].set_axis_off()
    axs1[0].set_title('Original')
    for i in range(tt):
        before = np.copy(ffLena)
        gg = sizes[i]
        hh = 2*aa-gg
        #hh = int(0.5*(aa+gg))
        mm = hh 
        # left
        before[:,:hh] += ffLena[:,gg:]
        # right
        before[:,gg:] += ffLena[:,:hh]
        # up
        before[:hh,:] += ffLena[gg:,:]
        # down
        before[gg:,:] += ffLena[:hh,:]
        # 4 corners
        before[gg:,gg:] += ffLena[:hh,:hh]
        before[gg:,:hh] += ffLena[:hh,gg:]
        before[:hh,gg:] += ffLena[gg:,:hh]
        before[:hh,:hh] += ffLena[gg:,gg:]
        
        aliased = np.fft.ifft2(np.fft.ifftshift(before)).real
        axs[i+1].imshow(aliased, cmap='gray')
        axs[i+1].set_axis_off()
        axs[i+1].set_title(f'{sizes[i]}')
        
        axs1[i+1].imshow(np.log(np.abs(before)), cmap='gray')
        axs1[i+1].set_axis_off()
        axs1[i+1].set_title(f'{sizes[i]}')
        print(f'convolve2d PSNR for sampling size {sizes[i]}: {PSNR(lena,aliased)}')
    fig.savefig('images/calias.png', bbox_inches='tight', pad_inches=0)
    fig1.savefig('images/cmag.png', bbox_inches='tight', pad_inches=0)


# %% Part D
def probD(file, sizes):
# Implement moving average filter of 3x3 and 7x7 filters
# scipy.signal.convolve2d
# Part D2
# Add random Gaussian noise
    image = img.imread(file)
    lena = np.asarray(image)
    xx, yy = lena.shape
    gaussian = np.random.normal(0, 10, (xx,yy)) 
    noisyLena = lena + gaussian
    
    tt = len(sizes)
    
    fig, axs = plt.subplots(1,tt+1, dpi=300)
    axs[0].imshow(lena, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('Original')
    for i in range(tt):
        ffilter = 1/sizes[i]**2 * np.ones((sizes[i],sizes[i]))
        cconv = scipy.signal.convolve2d(lena, ffilter, mode='same', boundary='wrap')
        axs[i+1].imshow(cconv, cmap='gray')
        axs[i+1].set_axis_off()
        axs[i+1].set_title(f'{sizes[i]}x{sizes[i]}')
        print(f'convolve2d PSNR for {sizes[i]}x{sizes[i]}: {PSNR(lena,cconv)}')
    fig.savefig('images/dlena.png',  bbox_inches='tight', pad_inches=0)
  
# Part D2
# Add random Gaussian noise
    fig1, axs1 = plt.subplots(1,tt+1, dpi=300)
    axs1[0].imshow(noisyLena, cmap='gray')
    axs1[0].set_axis_off()
    axs1[0].set_title('Noisy')
    print(f'Noisy lena PSNR: {PSNR(lena, noisyLena)}')
    for i in range(tt):
        ffilter = 1/sizes[i]**2 * np.ones((sizes[i],sizes[i]))
        cconv = scipy.signal.convolve2d(noisyLena, ffilter, mode='same', boundary='wrap')
        axs1[i+1].imshow(cconv, cmap='gray')
        axs1[i+1].set_axis_off()
        axs1[i+1].set_title(f'{sizes[i]}x{sizes[i]}')
        print(f'Noisy lena convolve2d PSNR for {sizes[i]}x{sizes[i]}: {PSNR(lena,cconv)}')
    fig1.savefig('images/dnoisy.png', bbox_inches='tight', pad_inches=0)    
    
# %% Main Function
if __name__ == "__main__":
    file = "lena512.bmp"
    probA(file)
    probB(file, [128, 32])
    probC(file, [384, 256])
    probD(file, [3, 7])