---

---

Prompt
---
For this mini-project, we will consider non-local means denoising, which was covered in [[Lecture13 - PatchProcessing_LowRankModels.pdf]] (see [[NonLocalMeans.pdf]]). As before, the mini-project will involve the following:

1. [ ] Read & understand this paper
2. [ ] Implementing the processing techniques proposed in the work in Python (or similar software), using either numerical or real-life datasets.
3. [ ] Submit a written report in IEEE conference format, along with the Python source files to re-generate the results. The report should include an abstract, and the following sections: Introduction, Methods, Results, and Discussion; should be approximately 4-pages in two-column format, excluding references. The Discussion section should identify the limitations of the technique (regardless of the success of your own Python implementation).

Given that this mini-project focuses on image restoration instead of image enhancement, the algorithm can be evaluated quantitatively and compared to others in the Results section. To this end, the expectation is the following:

1.  [ ] Choose a few standard images (e.g. cameraman, peppers, etc). Add Gaussian noise to these at different noise levels (as in Assignment 1). 
2. [ ] Apply the nonlocal means denoising algorithm. Calculate its PSNR and SSIM (you can use existing toolboxes). 
3. [ ] Apply comparison methods (e.g. moving average or median filter). Calculate their PSNR and SSIM.
4. [ ] Show example noisy & denoised images.
5. [ ] Report PSNRs for each method in a table for different Gaussian noise standard deviations. Repeat for SSIM.

This in general is how we report results for denoising/restoration/reconstruction algorithms. On a side note, we also usually report the influence of hyperparameters (e.g. search window, patch size, kernel weight parameter). These can also be quantified (for non-local means only in this case), but a detailed study of such quantification will not be feasible given the 4-page report. Nonetheless, you should tune these parameters prior to running the algorithm.

If you have questions about the procedures, please reach out to me and/or Merve. While the implementation is straightforward, and essentially only requires a search within a window followed by calculation of the weights, it is a good idea not to wait until close to the deadline to get started.

With the extension, this second mini-project is due on November 27 (11:59 pm).


---

[[NonLocalMeans.pdf]]

https://www.math.ucdavis.edu/~saito/data/acha.read.s11/buades-coll-morel-siamrev.pdf
---
section 4
http://mw.cmla.ens-cachan.fr/megawave/demo/
section 6

![[Pasted image 20231109202754.png]]
![[Pasted image 20231109202805.png]]