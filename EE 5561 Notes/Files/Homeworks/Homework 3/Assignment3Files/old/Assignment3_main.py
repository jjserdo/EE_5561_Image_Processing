from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from cg_solve import cg_solve
try:
    from skimage.restoration import denoise_tv_chambolle as TV_denoise
except ImportError:
    from skimage.filters import denoise_tv_chambolle as TV_denoise

# example usage of the functions:
# ----cg_solve------------------------------------------
#     x_recon = A^-1.b
#     no_iter: numver of iterations
#     x_recon = cg_solve(b,A,no_iter)
# ----TV_denoise------------------------------------------
#     weight: Weight of the TV penalty term
#     x_denoised = TV_denoise(x_noisy, weight)


# %% Load the image
I = plt.imread('cameraman.tif').astype(np.float64) # original image
y = loadmat('Assignment3_blurry_image.mat')['y']   # blurred image

# Blur Kernel
ksize = 9 
kernel = np.ones((ksize,ksize)) / ksize**2

[h, w] = I.shape
kernelimage = np.zeros((h,w))
kernelimage[0:ksize, 0:ksize] = np.copy(kernel)
fftkernel = np.fft.fft2(kernelimage)

sigm = np.sqrt(0.1)
alpha = np.sqrt(sigm**2/ np.max(np.abs(fftkernel)))

def H(x):
    return np.real(np.fft.ifft2(np.fft.fft2(x) * fftkernel))

def HT(x):
    return np.real(np.fft.ifft2(np.fft.fft2(x) * np.conj(fftkernel)))


# %% Here, implement proximal gradient
step_size = alpha**2/sigm**2


# %% Here, implement ADMM
