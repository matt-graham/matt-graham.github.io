<div class='title-background'>
  <div class='title-banner'>
    <h1 class='title-heading'> 
      Continuously tempered Hamiltonian Monte Carlo
    </h1>
  </div>
</div>

**Matt Graham &lt;[matt-graham.github.io](http://matt-graham.github.io)&gt;**  
*Joint work with Amos Storkey*

<img width='35%' src='images/informatics-logo.svg' />

---

### TL;DR

  * Simple tempered dynamics extension to Hamiltonian Monte Carlo using continuous temperature variable. <!-- .element: class="fragment" data-fragment-index="1" -->
  * Improves exploration of multimodal target distributions and allows estimation of normalising constants. <!-- .element: class="fragment" data-fragment-index="2" -->
  * Straightforward to use in existing HMC implementations. <!-- .element: class="fragment" data-fragment-index="3" -->

----

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" data-state="dim-bg" -->

### Task

Given a (unnormalised) target density on $\vct{x} \in \set{X} \subseteq \reals^D$ <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \tgtden{\vct{x}} \propto \exp\lpa-\phi(\vct{x})\rpa,  
\] <!-- .element: class="fragment" data-fragment-index="1" -->

how can we estimate expectations with respect to $\pi$  <!-- .element: class="fragment" data-fragment-index="2" -->

\[
  \mathbb{E}\_{\pi} \lsb f \rsb = \int\_{\set{X}} f(\vct{x}) \,\tgtden{\vct{x}} \,\dr\vct{x}
\] <!-- .element: class="fragment" data-fragment-index="2" -->

and the unknown normalising constant of the density <!-- .element: class="fragment" data-fragment-index="3" -->

\[
  Z = \int_{\set{X}} \exp\lpa-\phi(\vct{x})\rpa \,\dr\vct{x} ?
\] <!-- .element: class="fragment" data-fragment-index="3" -->

Note:

Specific problem I will be considering is performing approximate inference with high-dimensional densities with a potentially large number of separated modes. 

To be concrete with notation, the task I will be considering is, given a usually unnormalised density defined by a potential function $\phi$ over a $D$-dimensional real valued state space, *click* can we both compute expectations of functions $f$ with respect to the target *click* and further can we estimate the normalising constant of the density. 

----

<!-- .slide: data-background-image="images/2d-density-mcmc-1.svg" -->

  <div style='background-color: rgba(255, 255, 255, 0.3);'>
    <h3> 
      Markov chain Monte Carlo
    </h3>
  </div>

----

<!-- .slide: data-background-image="images/2d-density-mcmc-2.svg" -->

<div style='background-color: rgba(255, 255, 255, 0.3);'>
  <h3> 
    Markov chain Monte Carlo
  </h3>
</div>

---

<!-- .slide: data-background-video="images/2d-density-hmc.mp4" data-background-video-loop="true" data-state="dim-bg-video" -->

<h3 style='font-size: 110%;'>Hamiltonian Monte Carlo (HMC) <span class='ref'>Duane+ 1987</span></h3>

Auxiliary variable MCMC method - augments space with momentum. <!-- .element: class="fragment" data-fragment-index="1" -->

Simulate Hamiltonian dynamics in augmented space to generate proposed update. <!-- .element: class="fragment" data-fragment-index="2" -->

Accept or reject proposed update in Metropolis step. <!-- .element: class="fragment" data-fragment-index="3" -->

----

### Black-box inference with HMC

<div style='text-align: center; margin: auto;'>
  <div style='display: inline-block;'>
    <img src='images/stan-logo.png' height='150px' />
  </div>
  <div style='display: inline-block;'>
    <img src='images/pymc3-logo-cropped.png' height='150px' />
  </div>
</div>

  * Long-range moves in high-dimensional $\set{X}$. <!-- .element: class="fragment" data-fragment-index="1" -->

  * Adaptive: No U-Turns Sampler <span class='ref'>(Hoffman & Gelman 2014)</span>.
<!-- .element: class="fragment" data-fragment-index="2" -->

  * However: <!-- .element: class="fragment" data-fragment-index="3" -->
    *  Poor performance in multimodal targets. <!-- .element: class="fragment" data-fragment-index="4" -->
    *  Non-trivial to use samples to estimate $Z$. <!-- .element: class="fragment" data-fragment-index="4" -->

---

<!-- .slide: data-background-image="images/bimodal-geometric-bridge-visualisation.svg" data-background-size="auto 95%" data-state="dim-bg" -->

### Thermodynamic methods

Introduce inverse temperature $\beta$<!-- .element: class="fragment" data-fragment-index="1" -->

<p class="fragment" data-fragment-index="2">and simple normalised <em>base density</em> $\exp\lpa-\psi(\vct{x})\rpa$. </p>

\[
  p\lpa \vct{x} \gvn \beta \rpa \propto
  \exp\lpa -\beta \phi(\vct{x}) - (1 - \beta) \psi(\vct{x}) \rpa
\] <!-- .element: class="fragment" data-fragment-index="3" -->

----

<!-- .slide: data-background-image="images/bimodal-geometric-bridge-visualisation.svg" data-background-size="auto 95%" -->

---

<!-- .slide: data-background-image="images/annealed-importance-sampling-0.svg" data-background-size="auto 95%"  -->

<div style='background-color: rgba(255, 255, 255, 0.3);'>
<h3 style='font-size: 110%;'>Annealed importance sampling (AIS) <span class='ref'>(Neal 2001)</small></h3>
</div>

----

<!-- .slide: data-background-image="images/annealed-importance-sampling-1.svg" data-background-size="auto 95%" -->

----

<!-- .slide: data-background-image="images/annealed-importance-sampling-2.svg" data-background-size="auto 95%" -->

----

<!-- .slide: data-background-image="images/annealed-importance-sampling-3.svg" data-background-size="auto 95%" -->

----

<!-- .slide: data-background-image="images/annealed-importance-sampling-4.svg" data-background-size="auto 95%" -->

----

<!-- .slide: data-background-image="images/annealed-importance-sampling-5.svg" data-background-size="auto 95%" -->

---

<!-- .slide: data-background-image="images/simulated-tempering-0.svg" data-background-size="auto 95%"  -->

<div style='background-color: rgba(255, 255, 255, 0.3);'>
<h3 style='font-size: 110%;'>Simulated tempering (ST) <span class='ref'>Marinari &amp; Parisi 1992</span></h3>
</div>

----

<!-- .slide: data-background-image="images/simulated-tempering-1.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/simulated-tempering-2.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/simulated-tempering-3.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/simulated-tempering-4.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/simulated-tempering-5.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/simulated-tempering-6.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/rb-simulated-tempering-0.svg" data-background-size="auto 95%"  -->

<div style='background-color: rgba(255, 255, 255, 0.3);'>
<h3 style='font-size: 100%;'>Rao-Blackwellized tempered sampling <span class='ref'>Carlson+ 2016</span></h3>
</div>

---

### Continuous tempering

Use continuous inverse temperature variable $\beta \in [0,1] \Rightarrow$

Avoid need to choose inverse temperature 'ladder'.

\[
  p(\vct{x},\beta) \propto \exp\lpa-\beta \lpa \phi(\vct{x}) + \log\zeta\rpa - (1- \beta)\psi(\vct{x})\rpa
\]<!-- .element: class="fragment" data-fragment-index="1" -->


<p style='color: #888; font-size: 80%;' class="fragment" data-fragment-index="2">$\log \zeta \approx \log Z$ $\Rightarrow$ $\frac{p(\beta=1)}{p(\beta=0)} \approx 1$</p>


----

### Continuous tempering

Conditional density $\beta \gvn \vct{x}$ : truncated exponential

\[
  p(\beta\gvn\vct{x}) = \frac{\exp(-\beta \Delta(\vct{x}))\Delta(\vct{x})}{1- \exp(-\beta\Delta(\vct{x}))},
\]

\[
  \Delta(\vct{x}) = \phi(\vct{x}) + \log\zeta - \psi(\vct{x})
\]

Can generate independent samples from $p(\beta\gvn\vct{x})$.<!-- .element: class="fragment" data-fragment-index="1" -->


---

### Rao-Blackwellisation

If $\lbrace \vct{x}^{(s)} \rbrace_{s=1}^S$ samples from joint $p(\vct{x},\beta)$<!-- .element: class="fragment" data-fragment-index="1" -->

\[
  Z  \approx \sum_{s=1}^S \frac{p(\beta=1|\vct{x}^{(s)})}{p(\beta=0|\vct{x}^{(s)})} \,\zeta
\]<!-- .element: class="fragment" data-fragment-index="2" -->


\[
  \int\_{\set{X}} f(\vct{x})\,\pi(\vct{x})\,\dr\vct{x}  \approx \sum_{s=1}^S \frac{p(\beta=1|\vct{x}^{(s)}) \,f(\vct{x}^{(s)})}{p(\beta=1|\vct{x}^{(s)})}.
\]<!-- .element: class="fragment" data-fragment-index="3" -->


---

<!-- .slide: data-background-image="images/gibbs-continuous-tempering-0.svg" data-background-size="auto 95%"  -->

<div style='background-color: rgba(255, 255, 255, 0.3);'>
<h3 style='font-size: 110%;'>Gibbs continuous tempering (Gibbs-CT)</h3>
</div>

----

<!-- .slide: data-background-image="images/gibbs-continuous-tempering-1.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/gibbs-continuous-tempering-2.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/gibbs-continuous-tempering-3.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/gibbs-continuous-tempering-4.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/gibbs-continuous-tempering-5.svg" data-background-size="auto 95%"  -->

---

### Joint continuous tempering (joint-CT)

Jointly update $\beta$ and $\vct{x}$ with HMC? <!-- .element: class="fragment" data-fragment-index="1" -->


Reparameterise $\beta \in [0, 1] \to u \in \reals$ <!-- .element: class="fragment" data-fragment-index="2" -->

e.g. $\beta(u) = \frac{1}{1 + \exp(-u)}$ <!-- .element: class="fragment" data-fragment-index="3" -->

<div style='font-size: 90%;' class="fragment" data-fragment-index="4">
\[
  p(\vct{x},u) \propto \left|\pd{\beta}{u}\right| \exp\lpa\beta(u) \lpa \phi(\vct{x}) + \log\zeta \rpa - (1 - \beta(u)) \psi(\vct{x})\rpa 
\]
</div>


---

<!-- .slide: data-background-image="images/joint-continuous-tempering-0.svg" data-background-size="auto 95%"  -->

<div style='background-color: rgba(255, 255, 255, 0.3);'>
<h3 style='font-size: 110%;'>Joint continuous tempering (joint-CT)</h3>
</div>

----

<!-- .slide: data-background-image="images/joint-continuous-tempering-1.svg" data-background-size="auto 95%"  -->

----

<!-- .slide: data-background-image="images/joint-continuous-tempering-2.svg" data-background-size="auto 95%"  -->


----

<!-- .slide: data-background-image="images/joint-continuous-tempering-3.svg" data-background-size="auto 95%"  -->


----

<!-- .slide: data-background-image="images/joint-continuous-tempering-4.svg" data-background-size="auto 95%"  -->


---

<!-- .slide: data-background-image="images/fitting-base-distribution-0.svg"  data-state="dim-bg"-->

### Choosing a base density

Important in determining how flat $p(\beta)$ is.<!-- .element: class="fragment" data-fragment-index="1" -->

Ideally $\exp\lpa-\phi(\vct{x})\rpa \approx \zeta \exp\lpa-\psi(\vct{x})\rpa$.<!-- .element: class="fragment" data-fragment-index="2" -->

Smaller $\mathbb{D}_{\textrm{KL}}\lsb \exp\lpa-\psi(\vct{x})\rpa \,\Vert\, \frac{1}{Z}\exp\lpa-\phi(\vct{x})\rpa\rsb$ $\Rightarrow$ flatter $p(\beta)$.<!-- .element: class="fragment" data-fragment-index="3" -->


----

<!-- .slide: data-background-image="images/fitting-base-distribution-1.svg" -->

<div style='background-color: rgba(255, 255, 255, 0.3); margin-left:-500px; padding-left:500px; margin-right:-500px; padding-right:500px;'>
  <p> 
    $\exp\lpa-\psi(\vct{x})\rpa =$ prior?
  </p>
</div>

----

<!-- .slide: data-background-image="images/fitting-base-distribution-2.svg" -->

<div style='background-color: rgba(255, 255, 255, 0.5); margin-left:-500px; padding-left:500px; margin-right:-500px; padding-right:500px;'>
  <p> 
    $\exp\lpa-\psi(\vct{x})\rpa =$ Gaussian variational approximation?
  </p>
</div>

----

<!-- .slide: data-background-image="images/fitting-base-distribution-3.svg" -->

<div style='background-color: rgba(255, 255, 255, 0.5); margin-left:-500px; padding-left:500px; margin-right:-500px; padding-right:500px;'>
  <p> 
    $\exp\lpa-\psi(\vct{x})\rpa =$ mixture of approximations?
  </p>
</div>

----

<!-- .slide: data-background-image="images/fitting-base-distribution-4.svg" -->

<div style='background-color: rgba(255, 255, 255, 0.5); margin-left:-500px; padding-left:500px; margin-right:-500px; padding-right:500px;'>
  <p> 
    $\exp\lpa-\psi(\vct{x})\rpa =$ Gaussian moment-matched to mixture of approximations?
  </p>
</div>

---

<!-- .slide: data-background-video="images/20d-bmr-example-1.mp4" data-background-video-loop="true" -->

### Boltzmann machine relaxations 

<p class="fragment" data-fragment-index="1">Continuous relaxation of Boltzmann machine corresponding to structure Gaussian mixture model <span class='ref'>(Zhang+ 2012)</span>.</p>

10 generated frustrated 30-unit systems - highly multimodal.<!-- .element: class="fragment" data-fragment-index="2" -->

Ground truth for $Z$ and $\expc{\vct{x}}$ by exhaustive summation.<!-- .element: class="fragment" data-fragment-index="3" -->

Base density: Gaussian moment matched to mixture of mean-field approximations.<!-- .element: class="fragment" data-fragment-index="4" -->

----

### Boltzmann machine relaxation results

\[ \log Z \]

<img src='images/gaussian-bm-relaxation-30-unit-scale-6-log-norm-rmses.svg' width='80%' />

----

### Boltzmann machine relaxation results

\[ \expc{\vct{x}} \]

<img src='images/gaussian-bm-relaxation-30-unit-scale-6-mean-rmses.svg' width='80%' />

---

<!-- .slide: data-background-image="images/omniglot-samples.png" data-background-size="contain" data-state="dim-bg" -->

### OMNIGLOT IWAE marginal likelihood 

<p class="fragment" data-fragment-index="1">Importance weighted autoencoder <span class='ref'>(Burda+ 2016)</span> trained on binarised OMNIGLOT dataset.</p>

<p class="fragment" data-fragment-index="2">Estimate log marginal likelihood of 1000 *generated* images under trained decoder model (1000$\times$50 latent dimensions).</p>

<p class="fragment" data-fragment-index="3">Use bidirectional Monte Carlo  <span class='ref'>(Grosse+ 2015)</span> to stochastically upper/lower bound log marginal likelihood.</p>

<p class="fragment" data-fragment-index="4">Base density: Gaussian approximate latent posteriors from trained encoder model <span class='ref'>(Wu+ 2017)</span>.</p>

----

### IWAE marginal likelihood results

<img src='images/omni-marginal-likelihood-est.svg' width='90%' />

---

### Hierarchical regression model 

<p class="fragment" data-fragment-index="1">Hierarchical linear regression model applied to household Radon measurement dataset <span class='ref'>(Gelman & Hill 2006)</span>.</p>

<p class="fragment" data-fragment-index="2">919 data points and 92 free parameters.</p>

<img src='images/pymc3-logo.svg' height='200px' class="fragment" data-fragment-index="3" />

<p class="fragment" data-fragment-index="4">Use ADVI <span class='ref'>(Kucukelbir+ 2016)</span> to fit base density.</p>

<p class="fragment" data-fragment-index="4">Use NUTS <span class='ref'>(Hoffman & Gelman 2014)</span> in augmented space.</p>

----

### Hierarchical regression model results

<img src='images/hier-lin-regression-marg-lik.svg' width='90%' />

---

### Conclusions

  * Thermodynamic HMC augmentation which improves mode-hopping and allows estimation of $Z$.<!-- .element: class="fragment" data-fragment-index="1" -->
  * Easily used within existing HMC implementations.<!-- .element: class="fragment" data-fragment-index="2" -->
  * Exploits cheap deterministic approximations to $\pi(\vct{x})$ while still allowing asymptotic exactness.<!-- .element: class="fragment" data-fragment-index="3" -->

---

### Acknowledgements

<div>
   <div style='display: inline-block; padding: 10px;'>
     <img src='images/amos-storkey.jpg' height='150px' style='margin: 0;' />
     <div><small>Amos Storkey</small></div>
  </div>
</div>


<img src='images/informatics-logo.svg' width='35%'
 style='background: none; border: none; box-shadow: none;' />

<div style='display: inline-block;'>
   <img src='images/dtc-logo.svg' height='80px'
     style='vertical-align: middle; display: inline-block; background: none; border: none; box-shadow: none; margin: 10px;' />
   <div style='display: inline-block; width: 200px; vertical-align: middle; text-transform: uppercase; font-size: 35%;'>
       Doctoral Training Centre in Neuroinformatics and Computational Neuroscience
   </div> 
</div>

<div>
   <img src='images/epsrc-logo.svg' height='60px'
     style='background: none; border: none; box-shadow: none; margin: 10px;' />
   <img src='images/bbsrc-logo.svg' height='40px'
     style='background: none; border: none; box-shadow: none; margin: 10px;' />
   <img src='images/mrc-logo.svg' height='60px'
     style='background: none; border: none; box-shadow: none; margin: 10px;' /> 
</div>

---

## Thanks for listening. 
## Any questions?

<br />

<i class="fa fa-github fa-fw"></i> http://git.io/cthmc

---

### References

<ul style='font-size: 50%;'>
  
  <li>
  Hybrid Monte Carlo.  
  *Physics Letters B*, Duane, Kennedy, Pendleton & Roweth (1987).  
  </li>
  <li>
  The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo.  
  *JMLR*, Hoffman & Gelman (2014).  
  </li>
  <li>
  Annealed importance sampling.  
  *Statistics and Computing*, Neal (2001).  
  </li>
  <li>
  Simulated tempering: a new Monte Carlo scheme.  
  *Europhysics Letters*, Marinari and Parsi (1992).
  </li>
  <li>
  Partition functions from Rao-Blackwellized tempered sampling.   
  *ICML*, Carlson, Stinson, Pakman and Paninski (2016). 
  </li>
  <li>
  Continuous relaxations for discrete Hamiltonian Monte Carlo.   
  *NIPS*, Zhang, Ghahramani, Storkey and Sutton (2012).
  </li>
  <li>
  Importance weighted autoencoders   
  *ICLR*, Burda, Grosse and Salakhutdinov (2016).
  </li>
  <li>
  Sandwiching the marginal likelihood using bidirectional Monte Carlo.   
  *arXiv*, Grosse, Ghahramani and Adams (2015).
  <li>
  On the quantitative analysis of decoder-based generative models.   
  *ICLR*, Wu, Burda, Salakhutdinov and Grosse (2017).
  </li>
  <li>
  Data analysis using regression and multilevel/hierarchical models.  
  *Cambridge University Press*, Gelman and Hill (2006).
  </li>
  <li>Automatic differentiation variational inference.  
  *JMLR*, Kucukelbir, Tran, Ranganath, Gelman and Blei (2017).
  </li>
</ul>
