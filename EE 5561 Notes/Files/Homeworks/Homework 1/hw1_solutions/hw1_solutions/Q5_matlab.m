clear all;
%% functions
fft2c = @(x) fftshift(fftshift(fft2(ifftshift(ifftshift(x,1),2)),1),2);
ifft2c = @(x) ifftshift(ifftshift(ifft2(fftshift(fftshift(x,1),2)),1),2);
%% Q5.a
lena_img= double(imread('lena512.bmp'));
fft_lena=fft2c(lena_img);
m=log(abs(fft_lena));
p=angle(fft_lena);
figure; 
subplot(1,3,1); imagesc(lena_img); colormap gray; title('Lena')
subplot(1,3,2); imagesc(m); colormap gray; title('Magnitude')
subplot(1,3,3); imagesc(p); colormap gray; title('Phase')
truesize([512 512]); 

%% Q5.b
[Nx, Ny] = size(lena_img);
% 128 x 128 LPF
N1 = 128;
LPF_128 = padarray(ones(N1),[(Nx-N1)/2 (Ny-N1)/2],0,'both');
img_LPF_128 = ifft2c(fft_lena.*LPF_128);
psnr1 = psnr(lena_img,img_LPF_128);
%32 x 32 LPF
N2 = 32;
LPF_32 = padarray(ones(N2),[(Nx-N2)/2 (Ny-N2)/2],0,'both');
img_LPF_32 = ifft2c(fft_lena.*LPF_32);
psnr2 = psnr(lena_img,img_LPF_32);

figure; 
subplot(1,3,1); imagesc(lena_img); colormap gray; title('Original image')
subplot(1,3,2); imagesc(abs(img_LPF_128)); colormap gray; title(['Low Pass 128x128 - PSNR=', num2str(psnr1)])
subplot(1,3,3); imagesc(abs(img_LPF_32)); colormap gray; title(['Low Pass 32x32 - PSNR=', num2str(psnr2)])
truesize([512 512]);

%% Q5.c
[Nx, Ny] = size(lena_img);
% 128 x 128 LPF
N1 = 128;
HPF_128 = padarray(zeros(N1),[(Nx-N1)/2 (Ny-N1)/2],1,'both');
img_HPF_128 = ifft2c(fft_lena.*HPF_128);
psnr3 = psnr(lena_img,img_HPF_128);
%32 x 32 LPF
N2 = 32;
HPF_32 = padarray(zeros(N2),[(Nx-N2)/2 (Ny-N2)/2],1,'both');
img_HPF_32 = ifft2c(fft_lena.*HPF_32);
psnr4 = psnr(lena_img,img_HPF_32);

figure; 
subplot(1,3,1); imagesc(lena_img); colormap gray; title('Original image')
subplot(1,3,2); imagesc(abs(img_HPF_128)); colormap gray; title(['High Pass 128x128 - PSNR=', num2str(psnr3)])
subplot(1,3,3); imagesc(abs(img_HPF_32)); colormap gray; title(['High Pass 32x32 - PSNR=', num2str(psnr4)])
truesize([512 512]);

%% Q5.d
% 1/384 mm sampling
S1 = 384;
fft_sampled_384 = zeros(Nx+2*S1,Ny+2*S1);
aliase = padarray(fft_lena,[S1 S1],0);
for i=-1:1
    for j=-1:1
        fft_sampled_384 = fft_sampled_384 + circshift(aliase,S1*[i j]);
    end
end
fft_sampled_384 = fft_sampled_384(S1+1:S1+Nx,S1+1:S1+Ny);
img_sampled_384 = ifft2c(fft_sampled_384);

% 1/256 mm sampling
S2 = 256;
fft_sampled_256 = zeros(Nx+2*S2,Ny+2*S2);
aliase = padarray(fft_lena,[S2 S2],0);
for i=-1:1
    for j=-1:1
        fft_sampled_256 = fft_sampled_256 + circshift(aliase,S2*[i j]);
    end
end
fft_sampled_256 = fft_sampled_256(S2+1:S2+Nx,S2+1:S2+Ny);
img_sampled_256 = ifft2c(fft_sampled_256);

figure; 
subplot(1,3,1); imagesc(lena_img); colormap gray; title('Original image')
subplot(1,3,2); imagesc(abs(img_sampled_384)); colormap gray; title('Sampled with 1/384mm')
subplot(1,3,3); imagesc(abs(img_sampled_256)); colormap gray; title('Sampled with 1/256mm')
truesize([512 512]);

%% Q5.e
% without noise
M1 = 3;
kernel1 = ones(M1)/(M1^2);
img_maf_3 = imfilter(lena_img,kernel1);
psnr5 = psnr(lena_img,img_maf_3);

M2 = 7;
kernel2 = ones(M2)/(M2^2);
img_maf_7 = imfilter(lena_img,kernel2);
psnr6 = psnr(lena_img,img_maf_7);

figure; 
subplot(1,3,1); imagesc(lena_img); colormap gray; title('Original image')
subplot(1,3,2); imagesc(abs(img_maf_3)); colormap gray; title(['Moving Average Filter 3x3 - PSNR=', num2str(psnr5)])
subplot(1,3,3); imagesc(abs(img_maf_7)); colormap gray; title(['Moving Average Filter 7x7 - PSNR=', num2str(psnr6)])
truesize([512 512]);

% with Gaussian noise
sigma = 10;
noisy_img = lena_img + sigma*randn(size(lena_img))/256;

noisy_maf_3 = imfilter(noisy_img,kernel1);
psnr7 = psnr(lena_img,noisy_maf_3);

noisy_maf_7 = imfilter(noisy_img,kernel2);
psnr8 = psnr(lena_img,noisy_maf_7);

figure; 
subplot(1,3,1); imagesc(abs(noisy_img)); colormap gray; title('Noisy Image');
subplot(1,3,2); imagesc(abs(noisy_maf_3)); colormap gray; title(['Moving Average Filter 3x3 - PSNR=', num2str(psnr7)])
subplot(1,3,3); imagesc(abs(noisy_maf_7)); colormap gray; title(['Moving Average Filter 7x7 - PSNR=', num2str(psnr8)])
truesize([512 512]);

%% Functions
function [v] = psnr(I,In)
[m,n]=size(I);
value = max(I(:));
xmax=value(1);
v=10*log10(m*n*(xmax^2)/sum(sum(abs(I-In).^2)));
end

