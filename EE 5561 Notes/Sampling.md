September 12,  2023
W2D1 Tuesday
#lecture 

---

## [[Lecture03 - 2DFS_Sampling_2DDSSignals.pdf]]
Slide 22
- m and n are integers
- if parentheses then continuous, if brackets then discrete space
- band limited - means that the fourier transform is 0 outside the frequency
Slide 23
- sampling property 
Slide 24
- convolution of their fourier transforms
- F convolve gives you a shift and versions of itself
- assuming 1/dx is spaced apart enough so that it has no overlap
Slide 25
- need to be sampling fast 
- Nyquist rates; need to be 1/twicemaxFrequency
- think this is in [[HW 1 Notes]] as well
Slide 26
- to go back then just use a lowpass filter
- this is obvious to the EE people but not for the others
Slide 27
- the dark ones are in the high frequency region and now it's gonna look weird
Slide 28
- checkerboard artifacts due to aliasing
Slide 29
- [[HW 1 Notes]]
Slide 30
- in practice you don't use sinc interpolation because it's weird
- sample and hold has a motion blurring effect because your eye is expecting something 
