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

def A(x):
    return HT(H(x)) + rho*x

# %% Here, implement proximal gradient
step_size = alpha**2/sigm**2
# JJ's code starts here 
x = np.copy(y)
aTV = 1e-2
save = [500, 1000, 2500, 5000]
saves = []
for ind in range(7500):
    x_n = x + step_size * HT(y - H(x))
    x = TV_denoise(x_n, weight=aTV)
    if ind in save:
        saves.append(x)

fig, ax = plt.subplots(1,2, figsize=(6,6), dpi=150)
ax[0].imshow(y, cmap="gray")
ax[0].set_title("Blurry Image")
ax[1].imshow(x, cmap="gray")
ax[1].set_title("Deblurred Image")

fig, ax = plt.subplots(2,2, figsize=(8,8), dpi=150)
ax[0,0].imshow(saves[0], cmap="gray")
ax[0,0].set_title("500th Iteration")
ax[0,1].imshow(saves[1], cmap="gray")
ax[0,1].set_title("1000th Iteration")
ax[1,0].imshow(saves[2], cmap="gray")
ax[1,0].set_title("2500th Iteration")
ax[1,1].imshow(saves[3], cmap="gray")
ax[1,1].set_title("5000th Iteration")

# %% Here, implement ADMM
# JJ's code starts here
z = np.zeros_like(y)
u = np.zeros_like(y)
aTV = 1e-2
rho = 0.1
no_iter=20
save = [50, 100, 250]
saves = []
x = np.copy(z)
for ind in range(500):
    b = HT(y) + rho*(z-u)
    
    # cg_solve equivalent
    x0 = np.zeros_like(b)
    xcg = np.copy(x0)
    r = b - A(xcg)
    p = np.copy(r)
    rsold = np.sum(r * np.conj(r))
    for i in range(no_iter):
        Ap = A(p)
        alpha = rsold/np.sum(p * Ap)
        xcg = xcg + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * np.conj(r))
        p = r + rsnew / rsold * p
        rsold = np.copy(rsnew)
    x = xcg.copy()
    
    z = TV_denoise(x + u, weight=aTV)
    u += x - z
    if ind in save:
        saves.append(x)

fig, ax = plt.subplots(1,2, figsize=(6,6), dpi=150)
ax[0].imshow(y, cmap="gray")
ax[0].set_title("Blurry Image")
ax[1].imshow(x, cmap="gray")
ax[1].set_title("Deblurred Image")

fig, ax = plt.subplots(2,2, figsize=(8,8), dpi=150)
ax[0,0].imshow(saves[0], cmap="gray")
ax[0,0].set_title("50th Iteration")
ax[0,1].imshow(saves[1], cmap="gray")
ax[0,1].set_title("100th Iteration")
ax[1,0].imshow(saves[2], cmap="gray")
ax[1,0].set_title("250th Iteration")

