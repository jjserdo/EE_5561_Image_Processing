September 12,  2023
W2D1 Tuesday
#lecture 

---

## [[Lecture03 - 2DFS_Sampling_2DDSSignals.pdf]]
Slide 9
- LSI means linear shift invariant system
- arbitrary frequency u and v
- it's going to be a convolution of the input 
Slide 10
- that number $H(u,v)$ is the Fourier Transform
Slide 11
- eigenfunctions because it scales the input
- no midterms so not super critical :(
- in undergrad there was a sqrt because thinking in terms of angular
Slide 12 and 13
- this is in the [[HW 1 Notes]] 
- a box is a filter, a low-pass filter corresponds to a $sinc$ convolution
Slide 15 
- most important property is the convolution property
- DC value not usually spent time in undergrad
- The energy you calculate in the image space is the same as the energy in the fourier space [[Parseval's relattion]]; [[HW 1 Notes]]
Slide 17
- easier to look at the amplitude
- using log because the center is going to be really bright
- looks symmetric on the $y=-x$ diagonal because of the conjugate symmetry
- because this is discrete space then we can actually get a number for the energy
Slide 18
- 512 x 512; can low pass filter the middle; looks like it but a bit blurred out
- 16 x 16 filter; now it really looks blurred
- same image but losing some of the features but the contrast is still there
- inner's space gives you the contrast
- [[Gibb's Ringing Thing]] like thing, I feel like similar to the gibb's phenomenon and also smth from CFD
- [[HW 1 Notes]] This is the coding problem 
Missing Slide but is Slide 20 in the presentation
- outer space gets you the edges
- If we do the just the edges then we get the noise
- we want to get high frequency information well
- if shrink further we gets reverse ringing and get too many edges
Slide 20 
- can use Fourier Transform for anything
- Fourier Series k and l are integers
	- all of the energy is concentrated at those values
$$e^{i(2\pi / T_x kx + 2\pi / T_y ly)}$$
- Fourier Transform u and v are real numbers
- This is like the scaled version of the comb function
Slide 21 
- comb function in Fourier Space
