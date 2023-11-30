%% QUESTION 4
man = imread('man.png',1);
man = double(man)/255;
lena = imread('lena512.bmp');
lena = double(lena)/255;
coin = imread('eight.tif');
coin = double(coin)/255;

%% a) Combined Image
fft_lena = fftshift(fft2(ifftshift(lena)));
fft_man  = fftshift(fft2(ifftshift(man)));

fft_mix  = abs(fft_man).*exp(1i*angle(fft_lena));
img_mix  = fftshift(ifft2(ifftshift(fft_mix)));
img_mix  = abs(img_mix)/max(abs(img_mix(:)));

figure; 
subplot(1,3,1);
imshow(log(abs(fft_lena))/max(log(abs(fft_lena(:)))));
title('magnitude');
subplot(1,3,2);
imshow(angle(fft_lena)/max(angle(fft_lena(:))));
title('phase');
subplot(1,3,3);
imshow(lena);
title('Lena');
truesize([512 512]); 

figure; 
subplot(1,3,1);
imshow(log(abs(fft_man))/max(log(abs(fft_man(:)))));
title('magnitude');
subplot(1,3,2);
imshow(angle(fft_man)/max(angle(fft_man(:))));
title('phase');
subplot(1,3,3);
imshow(man);
title('Man');
truesize([512 512]); 

figure; 
subplot(1,3,1);
imshow(log(abs(fft_man))/max(log(abs(fft_man(:)))));
title('magnitude');
subplot(1,3,2);
imshow(angle(fft_lena)/max(angle(fft_lena(:))));
title('phase');
subplot(1,3,3);
imshow(img_mix);
title('Combined Image');
truesize([512 512]); 

%% b) DCT and DFT
% DCT
dct8  = @(img) dct_threshold(img.data);
lena_dct = blockproc(lena, [8 8], dct8);

figure;
subplot(1,2,1); imshow(lena);     title('Before DCT')
subplot(1,2,2); imshow(lena_dct); title('After DCT')
truesize([512 512]); 

% DFT
dft8  = @(I) dft_threshold(I.data);
lena_dft = blockproc(lena, [8 8], dft8);

figure;
subplot(1,2,1); imshow(lena);     title('Before DFT')
subplot(1,2,2); imshow(lena_dft); title('After DFT')
truesize([512 512]); 

%% c) Sharpening
h_lpf = ones(3)/9;
coin_lpf = imfilter(coin,h_lpf);
coin_hpf = coin-coin_lpf;
coin_sharp = coin + 2*coin_hpf;


figure;
subplot(2,2,1)
imshow(coin); title('original')
subplot(2,2,2)
imshow(coin_lpf); title('low pass')
subplot(2,2,3)
imagesc(coin_hpf); colormap gray; axis off; title('high pass')
subplot(2,2,4)
imshow(coin_sharp); title('sharpened')
truesize([242 308]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTIONS

function res = dct_threshold(img)
% dct for 8x8 blocks
DCT = dct2(img);
% thresholding dct
y = sort(abs(DCT),'descend');
thr = y(10);
DCT(abs(DCT)<thr) = 0;
% inverse dct
res = idct2(DCT);
end

function res = dft_threshold(img)
% dct for 8x8 blocks
DFT = fft2(img);
% thresholding dct
y = sort(abs(DFT),'descend');
thr = y(10);
DFT(abs(DFT)<thr) = 0;
% inverse dct
res = ifft2(DFT);
end

% median filter
function res = q5_part_a(I)
[Nx, Ny] = size(I);
res = I;
for x = 2:Nx-1
    for y = 2:Ny-1
        kernel = I(x-1:x+1,y-1:y+1);
        kernel = sort(kernel(:));
        res(x,y) = kernel(5);
    end
end
end

% regularized median filter
function res = q5_part_b(I,maxit)
[Nx, Ny] = size(I);
res = q5_part_a(I);
for t = 1:maxit
    res_pre = res;
    for x = 2:Nx-1
        for y = 2:Ny-1
            kernel_I = I(x-1:x+1,y-1:y+1);
            kernel_t = res_pre(x-1:x+1,y-1:y+1);
            kernel = [kernel_I; kernel_t];
            kernel = sort(kernel(:));
            res(x,y) = kernel(9+mod(t,2));
        end
    end
end
end
