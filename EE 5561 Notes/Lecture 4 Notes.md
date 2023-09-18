September 14, 2023
Week 2 Day 2
#lecture 
[[Lecture04 - 2DDS_Filters_DFT.pdf]]

---

Slide 4
- we are now in the discrete space
- easier to have the matrix representation of the filter
- center is at the 0 so that's why it is -1 to 1
Slide 5
- takes every other pixel
- fills it with 0s but make size bigger
Slide 6
- not sure about the BIBO stuff
Slide 7
- upsampling inverse is downsampling
Slide 8
- now with linear shift invariant systems
- PSF is the discrete space signal
Slide 10
- I have some notes of these from machine learning
Slide 17
- can increase value at the center to make it a bit better
- but noise will also be convected and magnified
Slide 20
- mirror extension
	- a non-zero padding but mirrored
- circular extension
	- copy from other side and nont mirrored
Systems in Frequency Domain
- noise is a flat spectrum
- noise reduction is a low-pass filter because center is high energy so we can just keep that, and remove the low frequencies
- edge enhancement its from high frequencies
LSI Systems in Frequency Domain
- 0 is low frequency
- -pi and pi are the high frequencies
- its a slowly decaying low-pass filter
Discrete Fourier Transform (DFT)
- usually half a lecture of your signals class at the very end

- phase is more important

EE 3015 - Signals and Systems openheim wilsky introduction to signals
EE 4541 - Digital Signal Processing 
Foundational notation 
	- parseval relation