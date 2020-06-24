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
</ul>
