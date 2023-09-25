September 19, 2023
Week 3 Day 1
#lecture 
[[Lecture05 - NoiseReduction_Sharpening_EdgeDetection.pdf]]

---

Slide 3
- can be countered by going to higher bits
Slide 5
- complement
- difference
- intersection
- union
Slide 6
- subtract from the max
Slide 7 
- good for fixing exposure issues
- simple contrast correction so it fixes the spread of the pixels
Slide 9
- image enhancement are somewhat subjective
- filter design
Slide 10
- different people might have different criteria so its gonna be good
- personal visual perception is not that good and it is easily fooled
Slide 12
- moving average blurs the original photo
Slide 14
- standard deviation 
- this noise is a gaussian distribution tho called white noise
- optical imaging lots of multiplicative noise
- cameras has additive noise
- if it is correlated noise then need to do something else called whitening
Slide 16
- assigns 28 to the center pixel and now moves to the next one
Slide 17
- good for keeping edges
- good for removing the high outlier
- outlier corrupts some parts
- edge corrupts some parts as well
Slide 18
- can see the dark pixels as dead sensors
- can see the white pixels as corrupted sensors
Slide 19
- can see the smearing effect
- the writings will be blurred out by the medians but almost got the full signal
Slide 21
- euclidean norm for l2
- dot product for l2
- manhattan distance is like the l1; lots of blocks
Slide 28
- derivative type operations also takes out the edges


---
Moving average filter - noise smoothing
derivative filters in x and y - ?
laplacian filter - sharpening
median filter - outlier 


Edge Detection
1. gaussian smoothing
2. derivative
3. find magnitude & orientation
4. extract edge points
5. linking & thresholding & hysteresis