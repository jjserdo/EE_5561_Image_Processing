clear all
I = double(imread('cameraman.tif'));
y = load('Assignment3_blurry_image.mat').y;

%% Blur Kernel
ksize = 9; kernel = ones(ksize) / ksize^2;

[h, w] = size(I); kernelimage = zeros(h,w);
kernelimage(1:ksize, 1:ksize) = kernel;
fftkernel = fft2(kernelimage);

sigm = sqrt(0.1);
alpha = sqrt(sigm^2/ max(abs(fftkernel(:))));

H = @(x) real(ifft2(fft2(x) .* fftkernel));
HT = @(x) real(ifft2(fft2(x) .* conj(fftkernel)));

%% Here, implement proximal gradient
step_size = alpha^2/sigm^2;


%% Here, implement ADMM
