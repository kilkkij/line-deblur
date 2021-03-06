# Line deblur
Remove linear camera shake.

# Theory

Using the Bayesian formulation and assuming independence of priors, the posterior probability of the latent image (I) and kernel (K) is given as a function of their prior probability densities and observation (Y).

P(I, K|Y) ~ P(Y|I, K) P(I) P(K)

For regularization, prior information is encoded as a function of various orders of discrete differentials approximated with convolutions. Informally:

P(I) ~ *N*(I - 0.5, σ<sub>0</sub>) *F*(∇I, σ<sub>1</sub>) *N*(∇²I, σ<sub>2</sub>)

P(K) ~ constant

P(Y|I, K) ~ *N*(convolution(I, K) - Y, σ<sub>obs</sub>)

The first order prior seems to be important in allowing sharpness while making sure flat regions are prioritized. That's why *F* is a mixture of two Cauchy distributions (see model.py).

Implicit in the above equations, observation standard deviation σ<sub>obs</sub> and first order difference σ<sub>1</sub> are hyperparameters with a gamma distributions.

The special feature of this solution is that the kernel is parameterized as a 2D vector, meaning that the camera shake is modeled as a convolution with a line kernel. The kernel parameters are inferred together with the image. (I can't recommend this approach -- the kernel parameter optimization seems slow and unstable.)

The estimate of the image is found in the maximum of the posterior.

Loosely based on, for example:
- Blind image deconvolution: theory and applications
- Shan et al. (2008) -- High-quality motion deblurring from a single image

# Examples

## Library without noise

A library with nice flat surfaces. Camera shake was generated.

![](examples/cases/library-nonoise/collage.png)

## Library with noise

Gaussian noise seems to make the model default to flat surfaces in the solution.

![](examples/cases/library/collage.png)

## Lena

Again, with noise and generated camera shake. There are hints of waviness, which I think happen because the mixture prior doesn't expect continuously varying smooth surfaces.

![](examples/cases/lena/collage.png)

## A real-life bike

Not bad, but the prior doesn't seem to punish white dots hard enough.

![](examples/cases/bike/collage.png)

## Conclusion

Tweaking the model is finicky. Particularly the prior distribution is difficult to have represent the statistics of real life photos. You might be better off using convolutional neural nets.