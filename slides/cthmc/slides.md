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

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" -->

---

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" data-state="dim-bg" -->

## Task


Given a (unnormalised) target density on $\vct{x} \in \set{X} \subseteq \reals^D$ <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \tgtden{\vct{x}} \propto \exp\lsb-\phi(\vct{x})\rsb,  
\] <!-- .element: class="fragment" data-fragment-index="1" -->

how can we compute expectations with respect to $\pi$  <!-- .element: class="fragment" data-fragment-index="2" -->

\[
  \mathbb{E}\_{\pi} \lsb f \rsb = \int\_{\set{X}} f(\vct{x}) \,\tgtden{\vct{x}} \,\dr\vct{x}
\] <!-- .element: class="fragment" data-fragment-index="2" -->

and the unknown normalising constant of the density <!-- .element: class="fragment" data-fragment-index="3" -->

\[
  Z = \int_{\set{X}} \exp\lsb-\phi(\vct{x})\rsb \,\dr\vct{x} ?
\] <!-- .element: class="fragment" data-fragment-index="3" -->

----

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" data-state="dim-bg" -->

e.g. Bayesian inference  <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \phantom{\pi\lpa\vct{x}\rpa =} 
  \pden{\vct{x} \gvn \vct{y}} = 
  \frac
  {\pden{\vct{y}\gvn\vct{x}}\pden{\vct{x}}}
  {\pden{\vct{y}}} 
  \phantom{= \frac{\exp\lsb -\phi(\vct{x})\rsb}{Z}}
\]  <!-- .element: class="fragment" data-fragment-index="2" -->

----

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" data-state="dim-bg" -->

e.g. Bayesian inference 

\[
  \pi\lpa\vct{x}\rpa = 
  \pden{\vct{x} \gvn \vct{y}} = 
  \frac
  {\pden{\vct{y}\gvn\vct{x}}\pden{\vct{x}}}
  {\pden{\vct{y}}} =
  \frac{\exp\lsb -\phi(\vct{x})\rsb}{Z}
\] 

----

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" data-state="dim-bg" -->

### Markov chain Monte Carlo (MCMC)

\[
  \phantom{
  \pi(\vct{x}') =
  \int_{\set{X}}
  }
  \,T(\vct{x}' \gvn \vct{x})
  \phantom{\,
  \pi(\vct{x})
  \,\dr\vct{x}
  }
\]

\[
  \phantom{\vct{x}^{(0)} \leftarrow p_0(\cdot)}
\]

\[
  \phantom{
  \vct{x}^{(t)} \leftarrow T(\cdot \gvn \vct{x}^{(t-1)})
  \quad \forall t \in \lbr 1 \dots T \rbr
  }
\]

----

<!-- .slide: data-background-image="images/2d-density-mcmc-0.svg" data-state="dim-bg" -->

### Markov chain Monte Carlo (MCMC)

\[
  \pi(\vct{x}') =
  \int_{\set{X}}
  T(\vct{x}' \gvn \vct{x}) \, \pi(\vct{x})
  \,\dr\vct{x}
\] 

\[
  \vct{x}^{(0)} \leftarrow p_0(\cdot)
\] <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \vct{x}^{(t)} \leftarrow T(\cdot \gvn \vct{x}^{(t-1)})
  \quad \forall t \in \lbr 1 \dots T \rbr
\] <!-- .element: class="fragment" data-fragment-index="2" -->

----

<!-- .slide: data-background-image="images/2d-density-mcmc-1.svg" -->

----

<!-- .slide: data-background-image="images/2d-density-mcmc-2.svg" -->

----

<!-- .slide: data-background-image="images/2d-density-mcmc-2.svg" data-state="dim-bg" -->

### Markov chain Monte Carlo (MCMC)

\[  
  \mathbb{E}\_{\pi} \lsb f \rsb \approx  \frac{1}{T} \sum_{t=1}^S \lsb f(\vct{x}^{(t)}) \rsb 
\] <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  T(\vct{x}'\gvn\vct{x}) = ?
\] <!-- .element: class="fragment" data-fragment-index="2" -->

\[
  Z = ?
\] <!-- .element: class="fragment" data-fragment-index="3" -->

---

<!-- .slide: data-background-video="images/2d-density-hmc.mp4" data-background-video-loop="true" data-state="dim-bg-video" -->

### Hybrid / Hamiltonian Monte Carlo (HMC) <small>Duane et al., 1987; Neal, 2011</small>


<span class="fragment" data-fragment-index="1">$\vct{x} \in \reals^D$</span><span class="fragment" data-fragment-index="2">
$\to (\vct{x},\,\vct{p}) \in \reals^D \times \reals^D$ </span>

\[
  \pi\lsb\vct{x},\,\vct{p}\rsb \propto 
  \exp \underbrace{
    \lsb  -\phi(\vct{x}) - \frac{1}{2}\vct{p}\tr\mtx{M}^{-1}\vct{p} \rsb
  }_{-H(\vct{x},\,\vct{p})}
\] <!-- .element: class="fragment" data-fragment-index="3" -->

\[
  \td{\vct{x}}{t} = \mtx{M}^{-1}\vct{p},
  \quad
  \td{\vct{p}}{t} = -\pd{\phi}{\vct{x}}
\] <!-- .element: class="fragment" data-fragment-index="4" -->

---


### Black-box HMC

  * Long-range moves in high-dimensional $\set{X}$. <!-- .element: class="fragment" data-fragment-index="1" -->

  * Target density must be differentiable. <!-- .element: class="fragment" data-fragment-index="2" -->

  * Gradients: reverse-mode automatic differentiation. <!-- .element: class="fragment" data-fragment-index="3" -->

  * Need to tune step-size $\delta t$ and integration time $\tau$. <!-- .element: class="fragment" data-fragment-index="4" -->

  * Adaptive: No U-Turns Sampler (NUTS) <small>Hoffman and Gelman, 2014.</small>
<!-- .element: class="fragment" data-fragment-index="5" -->

<div style='text-align: center; margin: auto;' class="fragment" data-fragment-index="6">
  <div style='display: inline-block;'>
    <img src='images/stan-logo.png' height='120px' />
    <div style="font-style: italic; font-family:'volkhov', serif;">Stan</div>
  </div>
  <div style='display: inline-block;'>
    <img src='images/pymc3-logo.jpg' height='200px' />
  </div>
</div>

---

### HMC in multimodal targets

<img src='images/hmc-bimodal-0.svg' width='80%' />

----

### HMC in multimodal targets

<img src='images/hmc-bimodal-1.svg' width='80%' />

----

### HMC in multimodal targets

<img src='images/hmc-bimodal-2.svg' width='80%' />

----

### HMC in multimodal targets

<img src='images/hmc-bimodal-3.svg' width='80%' />

----

### HMC in multimodal targets

<img src='images/hmc-bimodal-4.svg' width='80%' />

----

### HMC in multimodal targets

<img src='images/hmc-bimodal-5.svg' width='80%' />

----

### HMC in multimodal targets

<img src='images/hmc-bimodal-6.svg' width='80%' />

---

<!-- .slide: data-background-image="images/geometric-flat-bridge-beta-ensemble.svg" data-background-size="contain" data-state="dim-bg" -->

### Thermodynamic ensembles

\[
  \pi\lpa \vct{x} \gvn \beta \rpa =
  \frac{1}{\mathcal{Z}(\beta)}
  \exp\lsb -\beta \phi(\vct{x}) \rsb
\] <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \mathcal{Z}(\beta) = \int_{\set{X}} \exp\lsb -\beta \phi(\vct{x}) \rsb \,\dr\vct{x}
\] <!-- .element: class="fragment" data-fragment-index="2" -->

----

<!-- .slide: data-background-image="images/geometric-flat-bridge-beta-ensemble.svg" data-background-size="contain" -->

----

<!-- .slide: data-background-image="images/geometric-bridge-beta-ensemble.svg" data-background-size="contain" data-state="dim-bg" -->

### Thermodynamic ensembles

\[
  \pi\lpa \vct{x} \gvn \beta \rpa =
  \frac{1}{\mathcal{Z}(\beta)}
  \exp\lsb -\beta \phi(\vct{x}) - (1 - \beta) \psi(\vct{x}) \rsb
\] <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \mathcal{Z}(\beta) = \int_{\set{X}} \exp\lsb -\beta \phi(\vct{x}) - (1 - \beta) \psi(\vct{x}) \rsb \,\dr\vct{x}
\] <!-- .element: class="fragment" data-fragment-index="2" -->

----

<!-- .slide: data-background-image="images/geometric-bridge-beta-ensemble.svg" data-background-size="contain" -->

---

<!-- .slide: data-background-image="images/1d-gm-continuous-inv-temp-joint.svg" data-background-size="contain" -->

### Continuous inverse temperature $\beta$? 

----

<!-- .slide: data-background-image="images/1d-gm-adiabatic-monte-carlo-trajectory.svg" data-background-size="contain" data-state="dim-bg" -->

### Adiabatic Monte Carlo <small>Betancourt, 2014</small>

Flat target marginal $\pi(\beta) = 1$,  $\beta \in [0,\,1]$. <!-- .element: class="fragment current-visible" data-fragment-index="1" -->

\begin{align}
  \pi(\vct{x},\,\vct{p},\,\beta) 
  &=
  \pi(\vct{x} \gvn \beta) \pi(\beta) \pi(\vct{p})\\\\
  &=
  \exp\underbrace{\lsb 
    -\beta\phi(\vct{x}) - 
    \lpa 1 - \beta\rpa \psi(\vct{x}) -
    \frac{1}{2}\vct{p}\tr\mtx{M}^{-1}\vct{p} - 
    \log \mathcal{Z}(\beta) 
  \rsb}\_{H\_c(\vct{x},\,\vct{p},\,\beta)}
\end{align}<!-- .element: style="font-size:90%;" class="fragment" data-fragment-index="2" -->

\[
  \td{\vct{x}}{t} = \mtx{M}^{-1}\vct{p},~
  \td{\beta}{t} = -\vct{p}\tr \mtx{M}^{-1} \vct{p}
\]<!-- .element: style="font-size:90%;" class="fragment current-visible" data-fragment-index="3" -->

\[
  \td{\vct{p}}{t} = 
  -\beta \pd{\phi}{\vct{x}} - (1-\beta) \pd{\psi}{\vct{x}} +
  \lpa\phi(\vct{x}) - \psi(\vct{x}) + \color{red}{\pd{\log \mathcal{Z}}{\beta}}\rpa\vct{p}
\]<!-- .element:  style="font-size:90%;" class="fragment" data-fragment-index="4" -->

----

<!-- .slide: data-background-image="images/1d-gm-adiabatic-monte-carlo-stalled-trajectory.svg" data-background-size="contain" -->

---

### Extended Hamiltonian approach to continuous tempering <small>Gobbo and Leimkuhler, 2016</small>

\[
  \tilde{H}(\vct{x},\,u,\,\vct{p},\,v) =
  \beta(u) \phi(\vct{x}) + \omega(u) + \frac{1}{2}\vct{p}\tr\mtx{M}^{-1}\vct{p} + \frac{v^2}{2m}
\]<!-- .element: class="fragment" data-fragment-index="1" -->

<img src='images/inv-temp-control-func.svg' style='margin: 0; padding: 0;' width='70%' class="fragment" data-fragment-index="2" />

\[
  \pi\lsb\vct{x} \gvn {-\theta_1} \leq |u| \leq \theta_1\rsb \propto 
  \exp\lsb-\phi(\vct{x})\rsb
\]<!-- .element: class="fragment" data-fragment-index="3" -->


----

### Extended Hamiltonian approach to continuous tempering <small>Gobbo and Leimkuhler, 2016</small>

\[
  \tilde{H}(
    \color{green}{\underbrace{\vct{x},u}\_{\vct{\tilde{x}}}},\,
    \color{purple}{\underbrace{\vct{p},v}\_{\vct{\tilde{p}}}}
  ) =
  \color{green}{\underbrace{\\beta(u) \phi(\vct{x}) + \omega(u)}\_{\tilde{\phi}(\vct{\tilde{x}})}} + \color{purple}{\underbrace{\frac{1}{2}\vct{p}\tr\mtx{M}^{-1}\vct{p} + \frac{v^2}{2m}}\_{\frac{1}{2}\vct{\tilde{p}}\tr\mtx{\tilde{M}}^{-1}\vct{\tilde{p}}}}
\]

\[
  \tilde{H}(\vct{\tilde{x}},\,\vct{\tilde{p}}) = 
  \tilde{\phi}(\vct{\tilde{x}}) + \frac{1}{2}\vct{\tilde{p}}\tr\mtx{\tilde{M}}^{-1}\vct{\tilde{p}}
\]<!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \td{\vct{\tilde{x}}}{t} = \mtx{\tilde{M}}^{-1}\vct{\tilde{p}},
  \quad
  \td{\vct{\tilde{p}}}{t} = -\pd{\tilde\phi}{\vct{\tilde{x}}}
\]<!-- .element: class="fragment" data-fragment-index="2" -->

---

### Metadynamics <small>Laio and Parrinello, 2002</small>

\[
  \pi(u)
  \propto 
  \exp\lsb-\omega(u)\rsb
  \color{red}{\int\_{\set{X}} \exp\lsb-\beta(u)\phi(\vct{x})\rsb\,\dr\vct{x}}
\]<!-- .element: class="fragment current-visible" data-fragment-index="1" -->

<div class="fragment" data-fragment-index="2">
  <img src="/images/metadynamics.gif" width="40%" />
  $$u$$
  <small>Alessandro Laio, <a href='http://people.sissa.it/~laio/Research/Images/meta.gif'>http://people.sissa.it/~laio/Research/Images/meta.gif</a></small>
</div>

---

### Our approach

\begin{align}
  \tilde{H}(\vct{x},\,u,\,\vct{p},\,v) =\,&\, 
  \beta(u) \phi(\vct{x}) + \,
  \color{red}{\lsb 1 - \beta(u) \rsb \psi(\vct{x}) + 
  \beta(u) \log\zeta + \,} \\\\ \,&
  \frac{1}{2}\vct{p}\tr\mtx{M}^{-1}\vct{p} + \frac{v^2}{2m}
\end{align}<!-- .element: style="font-size:90%;" class="fragment" data-fragment-index="1" -->

$u \in [-1, +1]$ defined on circle.<!-- .element: class="fragment" data-fragment-index="2" -->

\[ 
  \log{\zeta} \approx \log Z
  \quad\textrm{and}\quad
  \exp\lsb -\psi(\vct{x}) \rsb
  \stackrel{\scriptscriptstyle\textrm{moments}}{\approx}
  \frac{1}{Z}\exp\lsb-\phi(\vct{x})\rsb
\]<!-- .element: class="fragment" data-fragment-index="3" -->

---

### Bounding the partition function

\[
  \pi(u) \propto \mathscr{Z}\lsb\beta(u)\rsb = \frac{\mathcal{Z}\lsb \beta(u)\rsb}{\zeta^{\beta(u)}}
\] <!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \mathcal{Z}\lsb \beta(u)\rsb =
  \int_{\set{X}} 
    \exp\lbr 
      \beta(u) \phi(\vct{x}) - \lsb 1 - \beta(u) \rsb \psi(\vct{x})
    \rbr
  \,\dr{\vct{x}}
\] <!-- .element: class="fragment current-visible" data-fragment-index="2" -->

\[
\beta(u) \lsb \log \frac{Z}{\zeta} - \mathbb{D}_{\rm KL}^{b\Vert t} \rsb \leq
\log\mathscr{Z}\lsb\beta(u)\rsb \leq
\beta(u) \log \frac{Z}{\zeta}
\] <!-- .element: class="fragment" data-fragment-index="3" -->

\[
  \mathbb{D}\_{\rm KL}^{b\Vert t} =
  \int\_{\set{X}}
    \exp\lsb-\psi(\vct{x})\rsb 
    \log \frac{\exp\lsb-\psi(\vct{x})\rsb }{\frac{1}{Z}\exp\lsb-\phi(\vct{x})\rsb }
  \,\dr\vct{x}
\] <!-- .element: class="fragment" data-fragment-index="4" -->

----

### Bounding the partition function

\[
  \pi(u) \propto \mathscr{Z}\lsb\beta(u)\rsb = \frac{\mathcal{Z}\lsb \beta(u)\rsb}{\zeta^{\beta(u)}}
\] 

\[
  \mathcal{Z}\lsb \beta(u)\rsb =
  \int_{\set{X}} 
    \exp\lbr 
      \beta(u) \phi(\vct{x}) - \lsb 1 - \beta(u) \rsb \psi(\vct{x})
    \rbr
  \,\dr{\vct{x}}
\] <!-- .element: class="fragment current-visible" data-fragment-index="-1" -->

\[
\beta(u) \log \frac{Z}{\zeta} - \lsb 1 - \beta(u) \rsb \mathbb{D}\_{\rm KL}^{t\Vert b} \leq
\log\mathscr{Z}\lsb\beta(u)\rsb
\] 

\[
  \mathbb{D}\_{\rm KL}^{t\Vert b} =
  \int\_{\set{X}}
    \frac{1}{Z}\exp\lsb-\phi(\vct{x})\rsb 
    \log \frac{\frac{1}{Z}\exp\lsb-\phi(\vct{x})\rsb }{\exp\lsb-\psi(\vct{x})\rsb }
  \,\dr\vct{x}
\] <!-- .element: class="fragment" data-fragment-index="1" -->

---

<!-- .slide: data-background-video="images/bmr-example-1.mp4" data-background-video-loop="true" data-state="dim-bg-video" -->

### Choosing a base distribution

Minimise $\mathbb{D}\_{\rm KL}^{b\Vert t}$(and/or $\mathbb{D}\_{\rm KL}^{t\Vert b}$),<!-- .element: class="fragment" data-fragment-index="1" -->

subject to $\exp\lsb-\psi(\vct{x})\rsb$ being a simple (unimodal) density.<!-- .element: class="fragment" data-fragment-index="2" -->

Choose parametric $\exp\lsb-\psi(\vct{x})\rsb$ (e.g. Gaussian) and minimise variational objective with respect to parameters?<!-- .element: class="fragment current-visible" data-fragment-index="3" -->

Iteratively locally match moments with expectation propagation?<!-- .element: class="fragment current-visible" data-fragment-index="4" -->

Fit a multiple local variational approximations and match moments of mixture of local approximations.<!-- .element: class="fragment" data-fragment-index="5" -->

---

<!-- .slide: data-background-image="images/3d-volume-ratio-example.svg" data-background-size="contain" data-state="dim-bg-2" -->

### Estimating $Z$

\[
  Z = 
  \frac{1 - \theta\_2}{\theta\_1}
  \frac{\mathbb{E}\_{\pi}\lsb \mathbb{1}[0 \leq |u| \leq \theta\_1]\rsb}
  {\mathbb{E}\_{\pi}\lsb\mathbb{1}[\theta\_2 \leq |u| \leq 1]\rsb}\zeta
\]<!-- .element: class="fragment current-visible" data-fragment-index="1" -->

\[
  Z \approx
  \frac{1 - \theta\_2}{\theta\_1}
  \frac
  {\sum\_{s=1}^S\lbr \mathbb{1}\lsb 0 \leq |u^{(s)}| \leq \theta\_1 \rsb\rbr}
  {\sum\_{s=1}^S\lbr \mathbb{1}\lsb \theta\_2 \leq |u^{(s)}| \leq 1 \rsb\rbr} \zeta
\]<!-- .element: class="fragment" data-fragment-index="2" -->

---

### 1D Gaussian mixture example

<div>
  <div style='display: inline-block; padding: 0px;'>
     <img src='images/extended-hamiltonian-1d-gm-x-chain-2.svg' height='470px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 0px;'>
    <img src='images/extended-hamiltonian-1d-gm-u-chain-2.svg' height='470px' style='margin: 0;' />
 </div>
</div>

---

<!-- .slide: data-background-video="images/bmr-example-1.mp4" data-background-video-loop="true" data-state="dim-bg-video" -->

### Boltzmann machine relaxation

\[
  \pi(\vct{s}) = 
  \frac{1}{Z_B} 
  \exp\lsb\frac{1}{2}\vct{s}\tr\mtx{W}\vct{s} + \vct{s}\tr\vct{b}\rsb
\]<!-- .element: class="fragment" data-fragment-index="1" -->

\[
  \pi(\vct{x}) \propto
  \exp\lsb-\frac{1}{2}\vct{x}\tr\vct{x} + \sum_{i=1}^{D_B}\log\cosh\lpa\vct{q}_i\tr\vct{x} + b_i \rpa\rsb
\]<!-- .element: class="fragment" data-fragment-index="2" -->

\[
  \mtx{W} + \mtx{D} = \mtx{Q}\mtx{Q}\tr
\]<!-- .element: class="fragment current-visible" data-fragment-index="3" -->

\[
  \log Z = \log Z_B + \frac{1}{2}\textrm{Tr}\lsb\mtx{D}\rsb + \frac{D}{2}\log(2\pi) - D_B \log(2)
\]<!-- .element: class="fragment" data-fragment-index="4" -->

\[
  \mathbb{E}\_\pi\lsb\vct{x}\rsb= \mtx{Q}\tr\mathbb{E}\_\pi\lsb\vct{s}\rsb
  \qquad
  \mathbb{E}\_\pi\lsb\vct{x}\vct{x}\tr\rsb = 
  \mtx{Q}\tr\mathbb{E}\_\pi\lsb \vct{s}\vct{s}\tr \rsb \mtx{Q} + \mtx{I}
\]<!-- .element: class="fragment" data-fragment-index="4" -->


----

### Boltzmann machine relaxation results

<div>
   <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-first-mom-err-2.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-second-mom-err-2.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-log-norm-const-err-2.svg' height='200px' style='margin: 0;' />
  </div>
</div>

<div>
   <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-first-mom-err-1.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-second-mom-err-1.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-log-norm-const-err-1.svg' height='200px' style='margin: 0;' />
  </div>
</div>

----

### Boltzmann machine relaxation results

<div>
   <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-first-mom-err-3.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-second-mom-err-3.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-log-norm-const-err-3.svg' height='200px' style='margin: 0;' />
  </div>
</div>

<div>
   <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-first-mom-err-4.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-second-mom-err-4.svg' height='200px' style='margin: 0;' />
  </div>
  <div style='display: inline-block; padding: 5px;'>
     <img src='images/bmr-20d-log-norm-const-err-4.svg' height='200px' style='margin: 0;' />
  </div>
</div>


---

### Conclusions

  * Thermodynamic HMC augmentation which improves mode-hopping and allows estimation of $Z$.<!-- .element: class="fragment" data-fragment-index="1" -->
  * Given $\zeta$ and $\psi$ can be easily used with existing HMC code.<!-- .element: class="fragment" data-fragment-index="2" -->
  * Exploits cheap deterministic approximations to $\pi(\vct{x})$ while still allowing asymptotic exactness.<!-- .element: class="fragment" data-fragment-index="3" -->

----

### On-going / future work

  * Non-geometric bridging between densities.
  * Exploiting multiple parallel chains.
  * $u$ dependent mass matrix / metric.
  * Using samples for intermediate $u$.

---

### Acknowledgements

<div>
   <div style='display: inline-block; padding: 10px;'>
     <img src='images/amos-storkey.jpg' height='150px' style='margin: 0;' />
     <div><small>Amos Storkey</small></div>
  </div>
  <div style='display: inline-block; padding: 10px;'>
    <img src='images/ben-leimkuhler.jpg' height='150px' style='margin: 0;' />
    <div><small>Ben Leimkuhler</small></div>
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

### References

<ul style='font-size: 65%;'>
  
  <li>
  Hybrid Monte Carlo. 
  *Physics Letters B*, Duane, Kennedy, Pendleton & Roweth (1987).  
  </li>
  <li>
  MCMC using Hamiltonian dynamics. 
  *Handbook of Markov Chain Monte Carlo*, Neal (2011).  
  </li>
  <li>
  The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. 
  *Journal of Machine Learning Research*, Hoffman & Gelman (2014).  
  </li>
  <li>
  Adiabatic Monte Carlo. 
  *arXiv preprint arXiv:1405.3489*, Betancourt (2014).  
  </li>
  <li>
  Extended Hamiltonian approach to continuous tempering. 
  *Physical Review E*, Gobbo & Leimkuhler (2015). 
  </li>
  <li>
  Escaping free-energy minima. 
  *Proceedings of the National Academy of Sciences*, Laio & Parrinello (2002).  
  </li>
</ul>
