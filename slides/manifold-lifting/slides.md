## <span style='color: #aaa;'>Manifold lifting:</span> Scaling MCMC to the vanishing noise regime

<img src='images/loop-manifold-sigma-5e-01.svg' height='300' />
<img src='images/loop-manifold-sigma-1e-01.svg' height='300' />
<img src='images/loop-manifold-sigma-2e-02.svg' height='300' />



### <span style='font-size: 90%;'>Joint work with Khai Xiang Au and Alex Thiery, National University of Singapore</span>


---

## Problem statement

**Task**: Infer a latent vector $\vct{\theta} \in \Theta \subseteq \mathbb{R}^{d_{\Theta}}$ given noisy observations $\vct{y} \in \set{Y} = \mathbb{R}^{d_{\set{Y}}}$ with

$$
  \vct{y} = F(\vct{\theta}) + \sigma \vct{\eta}
  \quad\text{and}\quad
  \vct{\eta} \sim \set{N}(\vct{0},\mathbf{I}_{d_{\set{Y}}}),
$$

where $F : \Theta \to \set{Y}$ generally non-linear and $\sigma > 0$.

----

## Problem statement

**Bayesian approach**: Assume prior beliefs specified by a distribution $\rho$  with density wrt $\lambda^{d_\Theta}$ on $\Theta$.  

Posterior distribution then<!-- .element: class="fragment fade-in" data-fragment-index="1" -->

$$
  \pi^\sigma(\dr\vct{\theta}) \propto
  \sigma^{-d_\set{Y}}
  \exp\big(
    {\textstyle-\frac{1}{2 \sigma^2}} \Vert\vct{y} -  F(\vct{\theta})\Vert^2
  \big) 
  \,\rho(\dr\vct{\theta}).
$$
<!-- .element: class="fragment fade-in" data-fragment-index="1" -->

<p class="fragment fade-in" data-fragment-index="2">**Aim**: design approximate inference method which remains efficient in $\sigma \to 0$ limit.</p>

---

## Toy 2D model

Running example with $d_{\Theta} = 2$ and $d_{\set{Y}} = 1$ and

$$ 
  \vct{\theta} \stackrel{\textrm{prior}}{\sim} \set{N}(\vct{0}, \mathbf{I}_{d_\Theta}), 
  ~~ 
  F(\vct{\theta}) = 
  \theta_1^2 + 3 \theta_0^2 \, (\theta_1^2 - 1),
  ~~
  y = 1.
$$

<img class="fragment fade-in" data-fragment-index="2" src='images/toy-2d-forward-op-pcolour.svg' />


---

## Posterior geometry as $\sigma \to 0$

In the vanishing noise limit posterior concentrates on $\set{S} = \lbrace \vct{\theta} \in \Theta : F(\vct{\theta}) = \vct{y} \rbrace$.<!-- .element: class="fragment fade-in" data-fragment-index="1" -->

<div class="third-column fragment fade-in" data-fragment-index="2">
  <img src='images/posterior-density-sigma-5e-01.svg' height='300' />
</div>
<div class="third-column fragment fade-in" data-fragment-index="3">
  <img src='images/posterior-density-sigma-1e-01.svg' height='300' />
</div>
<div class="third-column fragment fade-in" data-fragment-index="4">
  <img src='images/posterior-density-sigma-2e-02.svg' height='300' />
</div>

Strong anisotropy tangential and normal to $\set{S}$: challenging regime for MCMC methods.<!-- .element: class="fragment fade-in" data-fragment-index="5" -->

---

## MCMC performance in original space

Most general purpose MCMC algorithms on $\mathbb{R}^{d_\Theta}$ have a step size $\epsilon$ controlling scale of proposals.<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

How does acceptance probability vary with $\sigma$ &amp; $\epsilon$?<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="2" -->

<img class="fragment fade-in" data-fragment-index="3" src='images/loop-average-acceptance-rate-comparison-rwm-mala-hmc.svg' height='300'/>

----

## Manifold MCMC methods

<div style="position:relative; width:100%; height:120px;">
  <p class="fragment fade-in-then-out" data-fragment-index="0" style="width:100%; height: 100%; position:absolute;top:0;left:0;" >
    Anisotropic scaling of target distribution motivation behind Riemannian-manifold Langevin and HMC methods <small>(Girolami &amp; Calderhead, 2011; Xifara+, 2014)</small>.
  </p>
  <p class="fragment fade-in-then-out" data-fragment-index="1" style="width:100%; height: 100%; position:absolute;top:0;left:0;" >
    However when discretised with implicit leapfrog method performance still degenerates as $\sigma \to 0$.
  </p>
  <p class="fragment fade-in" data-fragment-index="2" style="width:100%; height: 100%; position:absolute;top:0;left:0;" >
    Instead of specifying manifold geometry intrinsically via metric, embed in extended space and use constrained leaprog integrator.
  </p>
</div>

<br />

<img class="fragment fade-in" data-fragment-index="1" src='images/loop-average-acceptance-rate-fisher-rm-hmc.svg' height='300'/>
<img class="fragment fade-in" data-fragment-index="3" src='images/loop-average-acceptance-rate-constrained-hmc.svg' height='300'/>

---

## Lifting the posterior distribution

Consider extended space $(\vct{\theta},\vct{\eta}) \in \Theta \times \set{Y}$ and define embedded manifold<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

$$
  \set{M}^\sigma = \lbrace (\vct{\theta},\vct{\eta}) \in \Theta \times \set{Y} : F(\vct{\theta}) + \sigma \vct{\eta} = \vct{y} \rbrace.
$$
<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

Posterior distribution $\pi^\sigma$ on $\Theta$ can be 'lifted' to distribution $\bar{\pi}^\sigma$ on to $\set{M}^\sigma$<!-- .element: class="fragment fade-in" data-fragment-index="2" -->

$$
  \bar{\pi}^\sigma(\dr\vct{\theta}, \dr\vct{\eta}) =
  \frac{
    \exp(-{\textstyle\frac{1}{2}\Vert\vct{\eta}\Vert^2})
    \frac{\dr\rho}{\dr\lambda^{d_\Theta}}(\vct{\theta})
  }{
  |\partial F(\vct{\theta})\partial F(\vct{\theta})\tr + \sigma^2\mathbf{I}_{d_{\set{Y}}}|^{\frac{1}{2}}
  }
  \,\set{H}^{d_\Theta}_{\set{M}^\sigma}(\dr\vct{\theta}, \dr\vct{\eta}).
$$
<!-- .element: class="fragment fade-in" data-fragment-index="2" -->

<!--

such that for all measurable $\varphi: \Theta \to \mathbb{R}$

$$
  \int_\Theta \varphi(\vct{\theta}) \,\pi^\sigma(\dr\vct{\theta}) =
  \int_{\set{M}^\sigma} \varphi(\vct{\theta}) \,\bar{\pi}^\sigma(\dr\vct{\theta},\dr\vct{\eta}) .
$$

-->

----

## Lifting the posterior distribution

Lifted distribution $\bar{\pi}^
\sigma$ (green) and original posterior $\pi^
\sigma$ (blue) for running example<!-- .element: class="fragment fade-in" data-fragment-index="1" -->

<div class="third-column fragment fade-in" data-fragment-index="2">
  <span style='font-size: 50%;'>$\sigma = 0.5$</span>
  <img src='images/loop-manifold-sigma-5e-01.svg' height='250' />
</div>
<div class="third-column fragment fade-in" data-fragment-index="3">
  <span style='font-size: 50%;'>$\sigma = 0.1$</span>
  <img src='images/loop-manifold-sigma-1e-01.svg' height='250' />
</div>
<div class="third-column fragment fade-in" data-fragment-index="4">
  <span style='font-size: 50%;'>$\sigma = 0.02$</span>
  <img src='images/loop-manifold-sigma-2e-02.svg' height='250' />
</div>

<p class="fragment fade-in" data-fragment-index="5">**Key point:** lifted posterior distribution $\bar{\pi}^
\sigma$ remains diffuse as $\sigma \to 0$.</p>

---

<h2 style="line-height: 80%;">Constrained Hamiltonian Monte Carlo <small>(Hartmann &amp; Schutte, 2005; Brubaker+, 2012; Leli&egrave;vre+, 2019)</small> </h2>

MCMC method based on simulating a constrained Hamiltonian dynamic defined by DAEs <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

$$
  \td{\vct{q}}{t} = \vct{p},
  ~~
  \td{\vct{p}}{t} = -\partial\phi(\vct{q})\tr + \partial C(\vct{q})\tr\vct{\lambda},
  ~~
  C(\vct{q}) = \vct{0},
$$ <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

<div style="position:relative; width:100%; height:120px;">
  <p class="fragment fade-in-then-out" data-fragment-index="2" style="width:100%; height: 100%; position:absolute;top:0;left:0;" >
    In our case: $\vct{q} = (\vct{\theta}, \vct{\eta})$, $C(\vct{\theta},\vct{\eta}) = F(\vct{\theta}) + \sigma\vct{\eta} - \vct{y}$ and <span style='font-size: 87%;'>$\phi(\vct{\theta},\vct{\eta}) = -\log\rho(\vct{\theta}) + {\textstyle \frac{1}{2}}\Vert\vct{\eta}\Vert^2 +  {\textstyle \frac{1}{2}}\log|\partial F(\vct{\theta})\partial F(\vct{\theta})\tr + \sigma^2\mathbf{I}|$</span>.
  </p>
  <p class="fragment fade-in-then-out" data-fragment-index="3" style="width:100%; height: 100%; position:absolute;top:0;left:0;" >
    Simulate using a constraint-preserving symplectic integrator such as RATTLE <small>(Andersen, 1983)</small>.
  </p>
  <p class="fragment fade-in" data-fragment-index="4" style="width:100%; height: 100%; position:absolute;top:0;left:0;">
    To enforce constraints in each step solve non-linear equations to project $\vct{q}$ on to manifold and linear equations to *project* $\vct{p}$ on to tangent space $\set{T}_\set{M}(\vct{q})$.
  </p>
</div>

----

## Constrained leapfrog step

<img height='400' src='images/constrained-step-cm-path.svg' />

---

## Manifold MCMC methods in Python

<img height='200' src='images/mici-logo-rectangular.svg' />

Available on Github at [git.io/mici.py](https://git.io/mici.py) or 

```pip install mici```


---

## Sampling efficiency

<div class='third-column fragment fade-in-then-semi-out' data-fragment-index="0">
  <p style='font-size: 60%; text-align: center;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Running 2D example</p>
  <img src='images/ess-loop.svg' height='250' />
</div>
<div class='third-column fragment fade-in' data-fragment-index="1">
  <p style='font-size: 60%; text-align: center;'>ODE parameter inference</p>
  <img src='images/ess-ode.svg' height='250' />
</div>
<div class='third-column fragment fade-in' data-fragment-index="1">
  <p style='font-size: 60%; text-align: center;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PDE inverse problem</p>
  <img src='images/ess-pde.svg' height='250' />
</div>

---

## Bayesian probabilistic numerical methods (BPNMs) <small>(Cockayne+, 2019)</small>

Latent variable $u \in \set{U}$ with prior distribution $\mu$ and  information operator $Y : \set{U} \to \set{Y}$. <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

Posterior $\mu^y$ is supported on $Y^{-1}(y)$ and satisfies for all $\mu$-measurable $\varphi : \set{U} \to \mathbb{R}$<!-- .element: class="fragment fade-in" data-fragment-index="2" -->

$$
 \int_{\set{U}} \varphi(u)\,\mu(\dr u) =
 \int_{\set{Y}} \int_{Y^{-1}(y)}\varphi(u)\,\mu^{y}(\dr u) \, (Y_\sharp\mu) (\dr y).
$$
<!-- .element: class="fragment fade-in" data-fragment-index="2" -->

---

## Numerical disintegration (ND)

Define family of relaxed posteriors for $\delta > 0$ <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

$$
  \mu^y_\delta(\dr u) \propto \delta^{-d_{\set{Y}}}\exp\left({\textstyle-\frac{1}{2\delta^2}}\Vert y - Y(u)\Vert^2 \right)\mu(\dr u).
$$
 <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

<p class="fragment fade-in-then-semi-out" data-fragment-index="2">Under appropriate assumptions as $\delta \to 0$, $\mu^y_\delta$ weakly converges to $\mu^y$ <small>(Cockayne+, 2019).</small></p>

Approximate $\mu^y$ by iteratively sampling from sequence of relaxed distributions $\lbrace \mu^y_{\delta_m} \rbrace_{m=1}^M$ with $\infty > \delta_1 > \delta_2 > \dots > \delta_M > 0$.<!-- .element: class="fragment fade-in" data-fragment-index="3" -->

---

## Sequential Monte Carlo (SMC) ND

<p class="fragment fade-in" data-fragment-index="1">**Input**: Markov kernels $K_{1:M}$ with $K_m \mu^y_{\delta_m} = \mu^y_{\delta_m}$</p>

$$
\begin{aligned}
&u^0_p \sim \mu \quad \forall p \in 1{:}P\\\
&\text{for } m \in 1{:}M:\\\
&\quad \tilde{u}^m_p \sim K_m(u^{m-1}_p, \cdot) & \forall p \in 1{:}P\\\  
&\quad w^m_p \gets \exp(-{\textstyle\frac{\delta_{m-1}-\delta_m}{2\delta_m\delta_{m-1}}}\Vert Y(u) - y\Vert^2) & \forall p \in 1{:}P \\\
&\quad u^m_p \sim \text{Discrete}(\tilde{u}^m_{1:P}, w^m_{1:P}) & \forall p \in 1{:}P
\end{aligned}
$$<!-- .element: class="fragment fade-in" data-fragment-index="2" -->

----

## Computational issues with SMC-ND

*'for small $\delta$, mixing of the chains tends to be poor'*

Need to use many MALA iterations for each kernel $K_{1:M}$ and large number of tolerances $M$. <!-- .element: class="fragment fade-in" data-fragment-index="1" -->

Also need to tune step size for each $K_{1:M}$. <!-- .element: class="fragment fade-in" data-fragment-index="2" -->

---

## <small>Idea 1:</small> Constrained HMC for BPNMs

Assuming $d_\set{U} < \infty$, $\mu \ll \lambda_{d_{\set{U}}}$, $Y$ is continuously differentiable and $\partial Y$ is full row-rank $\mu$-a.e. then <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

$$
  \mu^y(\dr u) \propto 
  \frac{\dr\mu}{\dr\lambda_{d_{\set{U}}}}(u)
  \det(\partial Y(u) \partial Y(u)\tr)^{-\frac{1}{2}}
  \,\set{H}^{d_{\set{U}}-d_{\set{Y}}}_{Y^{-1}(y)}(\dr u),
$$ <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

with support $Y^{-1}(y)$ in this case a (potentially disconnected) manifold embedded in $\set{U}$. <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

Use constrained HMC to directly construct Markov chain with $\mu^y$ as unique invariant distribution?<!-- .element: class="fragment fade-in" data-fragment-index="2" -->


---

## Potential issues

  * How to find point in $Y^{-1}(y)$ to initialise chain?<!-- .element: class="fragment fade-in" data-fragment-index="1" -->
  * $Y^{-1}(y)$ may consist of disconnected components $\rightarrow$ non-ergodic.<!-- .element: class="fragment fade-in" data-fragment-index="2" -->

---

## <small>Idea 2:</small> Constrained HMC within SMC-ND

Use constrained HMC for $K_{1:M}$ in SMC-ND method.<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

Earlier results $\Rightarrow$ sampling efficiency does not degenerate as $\delta \to 0$.<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="2" -->

Can use single fixed step size and also use dynamic integration time HMC implementation $\Rightarrow$ minimal tuning.<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="3" -->

<p class="fragment fade-in" data-fragment-index="4">**Technical issue**: is it possible to do a final move-reweight-resample to $\delta = 0$ 'exactly'?</p>

---

## References

<ul style="font-size: 70%;">
   <li>
     K. X. Au, M. M. Graham and A. H. Thiery. 
     Manifold lifting: scaling MCMC methods to the vanishing noise regime. 
     *arxiv:2003.03950*, 2020.
   </li>
   <li>
     J. Cockayne, C. J. Oates, T. J. Sullivan and M. Girolami. 
     Bayesian probabilistic numerical methods. 
     *SIAM Review*, 2019.
   </li>
   <li>
     M. Girolami and B. Calderhead. 
     Riemann manifold Langevin and Hamiltonian Monte Carlo methods. 
     *J-RSS (Series B)*, 2011.
   </li>
   <li>
     T. Xifara, C. Sherlock, S. Livingstone, S. Byrne and M. Girolami. 
     Langevin diffusions and the Metropolis-adjusted Langevin algorithm.  
     *Statistics &amp; Probability Letters*, 2014.
   </li>
</div>

----

## References

<ul style="font-size: 70%;">
   <li>
     H. C. Andersen. 
     RATTLE: A 'velocity' version of the SHAKE algorithm for molecular dynamics calculations. 
     *Journal of Comp. Physics*, 1983.
   </li>
   <li>
     C. Hartmann and C. Schutte. 
     A constrained hybrid Monte Carlo algorithm and the problem of calculating the free energy in several variables. 
     *ZAMM-Zeitschrift f&uuml;r Angewandte Mathematik*, 2005.
   </li>
   <li>
     M. A. Brubaker, M. Saelzmann and R. Urtasun. 
     A family of MCMC methods on implicitly defined manifolds. 
     *AISTATS*, 2012.
   </li>
   <li>
     T. Leli&egrave;vre, M. Rousset and G. Stoltz. 
     Hybrid Monte Carlo methods for sampling probability measures on submanifolds. 
     *Numerische Mathematik*, 2019.
   </li>
</div>
