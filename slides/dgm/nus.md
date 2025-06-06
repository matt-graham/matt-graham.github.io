
<h1 class='title-heading' style='font-size:140%;'> 
  Inference in differentiable <br /> generative models
</h1>

<img src='images/title-image-v2.svg' width='70%'
 style='background: none; border: none; box-shadow: none;' />
   
<p style='font-size: 90%; font-weight: bold;'>
  Matt Graham &lt;[matt-graham.github.io](http://matt-graham.github.io)&gt;
</p>

<p style='font-size: 80%; font-style: italic;'>
  Joint work with Amos Storkey
</p>

<img width='35%' src='images/informatics-logo.svg' />


---

### Problem description

<div class="fragment" data-fragment-index="0">

*Given:* Probabilistic model of 

<p>
    $\observed{\rvct{x}}$ <span class="observed">: observed variables $\in \observed{\set{X}}$,</span>
</p>
<p>
    $\latent{\rvct{z}}$ <span class="latent">: latent variables $\in \latent{\set{Z}}$.</span>
</p>

</div>

<p class="fragment" data-fragment-index="2">
  *Task:* estimate conditional expectations
</p>

$$\expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}}.$$ <!-- .element: class="fragment" data-fragment-index="2" -->

---

### Approximate inference

$$\expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} = \int\_{\latent{\set{Z}}} f(\latent{\vct{z}})\,\frac{\pden{\latent{\rvct{z}},\observed{\rvct{x}}}(\latent{\vct{z}},\observed{\vct{x}})}{\pden{\observed{\rvct{x}}}(\observed{\vct{x}})}\,\dr\latent{\vct{z}}$$ <!-- .element: class="fragment current-visible" data-fragment-index="0" -->

$$ \pi(\latent{\vct{z}}) = \pden{\latent{\rvct{z}},\observed{\rvct{x}}}(\latent{\vct{z}},\observed{\vct{x}}) \implies \expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} \propto \int\_{\latent{\set{Z}}} f(\latent{\vct{z}})\, \pi(\latent{\vct{z}})\,\dr\latent{\vct{z}}$$ <!-- .element: class="fragment" data-fragment-index="1" -->

<div>

<div class='half-column fragment' style='font-size: 90%;' data-fragment-index='2'>
<p>*Markov chain Monte Carlo*</p>

  $$ \pi(\latent{\vct{z}}) = \int\_{\latent{\set{Z}}} \mathsf{T}(\latent{\vct{z}}|\latent{\vct{z}'})\,\pi(\latent{\vct{z}'})\,\dr\latent{\vct{z}'}$$
  
  $$\expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} \approx \frac{1}{S}\sum\_{s=1}^S f(\latent{\vct{z}^{(s)}})$$ 

</div>

<div class='half-column fragment' style='font-size: 90%;' data-fragment-index='3'>
<p>*Variational inference*</p>

$$ \min\_{\vct{\theta}} \mathbb{D}\_{\textrm{KL}}\lsb q\_{\vct{\theta}} \Vert \pi\rsb$$ <br />

$$\expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} \approx \\\\ \int\_{\latent{\set{Z}}} f(\latent{\vct{z}})\,q\_{\vct{\theta}} (\latent{\vct{z}})\,\dr\latent{\vct{z}}$$

</div>

<div style='clear: both;'></div>

</div>

---

### Generative models

Probabilistic models specified by a generative process.

<img src='images/lotka-volterra-generator-samples.svg' width='20%'
 style='background: none; border: none; box-shadow: none;' class="fragment" data-fragment-index="1"/>
<img src='images/mnist-generator-samples.svg' width='20%'
 style='background: none; border: none; box-shadow: none;' class="fragment" data-fragment-index="1"/>
<img src='images/pose-generator-samples.svg' width='20%'
 style='background: none; border: none; box-shadow: none;' class="fragment" data-fragment-index="1"/>

<div  class="fragment" data-fragment-index="1">
 
Examples:

<ul>
  <li> Simulators of physical and biological processes. </li>
  <li> Models specified by differentiable networks (GANs, VAEs). </li>
  
</ul>

</div>

Often $\pden{\observed{\rvct{x}},\latent{\rvct{z}}}$ not explicitly defined. How to perform inference? <!-- .element: class="fragment" data-fragment-index="2" -->

---

### Generative models as transformations

<img src='images/title-image-v2.svg' width='75%'
 style='background: none; border: none; box-shadow: none;' />

Most (all?) generative models can be expressed in the form 

$$
  \input{\rvct{u}} \sim \rho
  \qquad
  \latent{\rvct{z}} = \vctfunc{g}\_{\latent{\rvct{z}}}(\input{\rvct{u}})
  \qquad
  \observed{\rvct{x}} = \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\rvct{u}})
$$

<ul>
<li class="fragment" data-fragment-index="1">
  $\rho$ is density of distribution of *random inputs* $\input{\rvct{u}} \in \input{\set{U}}$,
</li>
<li class="fragment" data-fragment-index="2"> $\vctfunc{g}\_{\latent{\rvct{z}}} : \input{\set{U}} \to \latent{\set{Z}}$ and $\vctfunc{g}\_{\observed{\rvct{x}}} : \input{\set{U}} \to \observed{\set{X}}$ are *generator functions*. </li>
</ul>

----

### Directed and undirected models

<div>

<div class='half-column fragment' data-fragment-index='1'>
<p>*Undirected model*</p>

<div class='img-row'>
<img src='images/undirected-generative-model.svg' height='200px' /> 
</div>

\begin{align}
  \latent{\rvct{z}} &= \vctfunc{g}\_{\latent{\rvct{z}}}(\input{\rvct{u}})\\\\
  \observed{\rvct{x}} &= \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\rvct{u}})
\end{align}
</div>

<div class='half-column fragment' data-fragment-index='2'>
<p>*Directed model*</p>

<div class='img-row'>
<img src='images/directed-generative-model.svg' height='200px' />
</div>

\begin{align}
  \latent{\rvct{z}} &= \vctfunc{g}\_{\latent{\rvct{z}}}(\input{\rvct{u}\_1})\\\\
  \observed{\rvct{x}} &= \vctfunc{g}\_{\observed{\rvct{x}}|\latent{\rvct{z}}}(\input{\rvct{u}\_2},\, \latent{\rvct{z}})
\end{align}
</div>

<div style='clear: both;'></div>

</div>

---

### Differentiable generative models

Concentrate on restricted case where

  * Variables real-valued $\input{\set{U}} \subseteq \reals^{D\_{\input{\rvct{u}}}}$, $\observed{\set{X}} \subseteq \reals^{D\_{\observed{\rvct{x}}}}$, $\latent{\set{Z}} \subseteq \reals^{D\_{\latent{\rvct{z}}}}$ <!-- .element: class="fragment" data-fragment-index="1" -->
  * Input density gradient $\pd{\rho}{\input{\vct{u}}}$ exists almost everywhere, <!-- .element: class="fragment" data-fragment-index="2" -->
  * Generator Jacobian $\pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}$ exists almost everywhere. <!-- .element: class="fragment" data-fragment-index="3" -->

For $\pden{\observed{\rvct{x}}}$ to be defined with respect to Lebesgue measure also require $D\_{\input{\rvct{u}}} \geq D\_{\observed{\rvct{x}}}$ and $\pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}$ full row-rank almost everywhere. <!-- .element: class="fragment" data-fragment-index="4" -->

---

### Example: MNIST Variational Autoencoder decoder <small>Kingma and Welling, 2013</small>

<div class='fragment' data-fragment-index='1' style='padding-bottom: 1em;'>
$\input{\rvct{u} = [\rvct{u}\_1;\, \rvct{u}\_2]}$ with $\rho(\input{\vct{u}}) = \mathcal{N}\lpa\input{\vct{u}};\,\vct{0},\,\mtx{I}\rpa$

$$
  \observed{\rvct{x}} = 
  \vctfunc{m}(\input{\rvct{u}_1}) + 
  \vctfunc{s}(\input{\rvct{u}_1}) \odot \input{\rvct{u}_2}
$$

</div>

<video autoplay loop class="fragment" data-fragment-index="2" width="720">
  <source data-src="images/mnist-prior.mp4" type="video/mp4" />
</video>

----

### Example: Pose projection generator

$\input{\rvct{u} = [\rvct{u}\_a;\, \rvct{u}\_b;\, \rvct{u}\_c;\, \rvct{u}\_1 \,\dots\, \rvct{u}\_J]}$ with $\rho(\input{\vct{u}}) = \mathcal{N}\lpa\input{\vct{u}};\,\vct{0},\,\mtx{I}\rpa$

$$
  \overset
  {\textrm{joint angles}}
  {\latent{\rvct{z}\_{a}} = \vctfunc{f}\_a(\input{\rvct{u}\_a})}
  \qquad
  \overset
  {\textrm{bone lengths}}
  {\latent{\rvct{z}\_{b}} = \vctfunc{f}\_b(\input{\rvct{u}\_b})}
  \qquad
  \overset
  {\textrm{camera parameters}}
  {\latent{\rvct{z}\_{c}} = \vctfunc{f}\_c(\input{\rvct{u}\_c})}
$$

$$
  \overset
  {\textrm{2D proj.}}
  {\observed{\rvct{x}_j}} = 
  \overset
  {\textrm{camera matrix}}
  {\mtxfunc{C}\lpa\latent{\rvct{z}_c}\rpa}
  \overset
  {\textrm{3D pos.}}{
  \vctfunc{r}_j\lpa
    \latent{\rvct{z}_a},\,
    \latent{\rvct{z}_b}
  \rpa} + 
  \overset
  {\textrm{obs. noise}}
  {\sigma\,\input{\rvct{u}_j}} 
  \quad\forall j \in \lbrace 1 \dots J \rbrace
$$ <!-- .element: class="fragment" data-fragment-index="2" -->

<video autoplay loop width="730" class="fragment" data-fragment-index="3">
  <source data-src="images/pose-prior-2.mp4" type="video/mp4" />
</video>


---

### Simulator models

Many simulators with continuous outputs can be expressed as directed differentiable generative models. <!-- .element: class="fragment" data-fragment-index="1" -->

Usually defined procedurally in code:<!-- .element: class="fragment" data-fragment-index="2" -->

```Python
def generator(rng):
    z = sample_from_prior(rng)
    x = simulate(z, rng)
    return x, z
```
<!-- .element: class="fragment" data-fragment-index="2" -->

----

### Example: Lotka-Volterra model

<img src='images/rabbit.svg' width='15%'
 style='vertical-align:middle; background: none; border: none; box-shadow: none;' />
 <img src='images/fox.svg' width='15%'
 style='vertical-align:middle; background: none; border: none; box-shadow: none;' />

Continuous variant of model of prey ($\observed{x_1}$) and predator ($\observed{x_2}$) populations

$$
    \textrm{d} \observed{x_1} = 
    (\latent{z_1} \observed{x_1} - \latent{z_2} \observed{x_1 x_2}) \textrm{d} t + 
    \textrm{d} n_1
$$ <!-- .element: class="fragment" data-fragment-index="1" -->

$$
    \textrm{d} \observed{x_2} = 
    (-\latent{z_3} \observed{x_2} + \latent{z_4} \observed{x_1 x_2}) \textrm{d} t + 
    \textrm{d} n_2
$$ <!-- .element: class="fragment" data-fragment-index="1" -->

where $n_1$ and $n_2$ are white noise processes. <!-- .element: class="fragment" data-fragment-index="1" -->

----

### Example: Lotka-Volterra model

Simulate at $T$ discrete time-steps

```Python
def sample_from_prior(rng):
    return np.exp(rng.normal(size=4) - mu)
    
def simulate(z, rng):
    x1_seq, x2_seq = [], []
    x1, x2 = x1_init, x2_init
    for t in range(T):
        x1 += ( z[0]*x1 - z[1]*x2) * dt + rng.normal()*dt**0.5
        x2 += (-z[2]*x2 + z[3]*x1) * dt + rng.normal()*dt**0.5
        x1_seq.append(x1)
        x2_seq.append(x2)
    return np.array(x1_seq), np.array(x2_seq)
```
<!-- .element: class="fragment" data-fragment-index="1" -->


$$
    \input{\rvct{u}} = 
    \lsb \input{\textrm{random number generator draws}} \rsb
$$  <!-- .element: class="fragment" data-fragment-index="2" -->

$$
    \observed{\rvct{x}} = 
    \lsb 
      \observed{
        \rvar{x}^{(1)}_1,\,\rvar{x}^{(1)}_2,
        \,\dots\,
        \rvar{x}^{(T)}_1,\,\rvar{x}^{(T)}_2
      }
   \rsb,
   \quad
   \latent{\rvct{z}} =
   \lsb
      \latent{
        \rvar{z_1},\,\rvar{z_2},\,\rvar{z_3},\,\rvar{z_4}
      } 
    \rsb
$$ <!-- .element: class="fragment" data-fragment-index="2" -->

----

### Example: Lotka-Volterra model

<video autoplay loop width="1080">
  <source data-src="images/lotka-volterra-prior.mp4" type="video/mp4" />
</video>

---

### Calculating derivatives


How do we propagate derivatives through complex generative models / simulators?

<div class="fragment" data-fragment-index="1">
  <p>Reverse mode automatic differentiation</p>
  <img src='images/theano-logo.svg' style="vertical-align: middle;" width='20%' />
  <img src='images/tensorflow-logo.svg' style="vertical-align: middle;" width='20%' />
  <img src='images/stan-logo.svg' style="vertical-align: middle; padding-left: 20px; padding-right: 20px;" width='12%' />
  <img src='images/pytorch-logo.svg' style="vertical-align: middle;" width='20%' />
</div>

Calculate $\pd{\rho}{\input{\vct{u}}}$ with cost $\sim$ evaluation of $\rho$ and  $\pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}$ with cost $\sim D\_{\observed{\rvct{x}}} \,\times$ cost of evaluating $\vctfunc{g}\_{\observed{\rvct{x}}}$. <!-- .element: class="fragment" data-fragment-index="2" -->

---

<!-- .slide: data-transition="none" -->
<h3 style='font-size: 120%;'>Inference in generative models?</h3>

<img src='images/abc-in-input-space-0-v2.svg' width='100%' /> 

----

<!-- .slide: data-transition="none" -->
<h3 style='font-size: 120%;'> Approximate Bayesian Computation (ABC) </h3>

<img src='images/abc-in-input-space-1-v2.svg' width='100%' /> 

----

<!-- .slide: data-transition="none" -->
<h3 style='font-size: 120%;'> Approximate Bayesian Computation (ABC) </h3>

<img src='images/abc-in-input-space-epsilon-1e-01-v2.svg' width='100%' /> 

----

<!-- .slide: data-transition="none" -->
<h3 style='font-size: 120%;'> Approximate Bayesian Computation (ABC) </h3>

<img src='images/abc-in-input-space-epsilon-5e-02-v2.svg' width='100%' /> 

----

<!-- .slide: data-transition="none" -->
<h3 style='font-size: 120%;'> Approximate Bayesian Computation (ABC) </h3>

<img src='images/abc-in-input-space-epsilon-3e-02-v2.svg' width='100%' /> 

---

### Problems with ABC approach

  * Curse of dimensionality <!-- .element: class="fragment" data-fragment-index="1" -->
    * Reduce $D\_{\observed{\rvct{x}}}$ with summary statistics $\observed{\rvct{s}} = \vct{s}(\observed{\rvct{x}})$ <!-- .element: class="fragment" data-fragment-index="2" -->
  * Multiple sources of approximation error <!-- .element: class="fragment" data-fragment-index="3" -->
    * Monte Carlo error ($\to 0$ as sample size $\to \infty$) <!-- .element: class="fragment" data-fragment-index="4" -->
    * Use of tolerance ($\to 0$ as $\epsilon \to 0$)  <!-- .element: class="fragment" data-fragment-index="5" -->
    * Use of summaries (0 only if $\vct{s}$ sufficient) <!-- .element: class="fragment" data-fragment-index="6" -->

---

<!-- .slide: data-transition="none" -->
### (Pseudo-marginal) ABC MCMC

Perturbatively update $\latent{\rvct{z}}$, independently sample $\observed{\rvct{x}}\gvn\latent{\rvct{z}}$

<img src='images/toy-example-pm-abc-mcmc-1.svg' width='80%' /> 

Can improve efficiency for high $D\_{\latent{\rvct{z}}}$.  
However prone to sticking pathologies. <!-- .element: class="fragment" data-fragment-index="1" -->

----

<!-- .slide: data-transition="none" -->
### (Pseudo-marginal) ABC MCMC

Perturbatively update $\latent{\rvct{z}}$, independently sample $\observed{\rvct{x}}\gvn\latent{\rvct{z}}$

<img src='images/toy-example-pm-abc-mcmc-2.svg' width='80%' /> 

Can improve efficiency for high $D\_{\latent{\rvct{z}}}$.  
However prone to sticking pathologies.

----

<!-- .slide: data-transition="none" -->
### (Pseudo-marginal) ABC MCMC

Perturbatively update $\latent{\rvct{z}}$, independently sample $\observed{\rvct{x}}\gvn\latent{\rvct{z}}$

<img src='images/toy-example-pm-abc-mcmc-3.svg' width='80%' /> 

Can improve efficiency for high $D\_{\latent{\rvct{z}}}$.  
However prone to sticking pathologies.

----

<!-- .slide: data-transition="none" -->
### (Pseudo-marginal) ABC MCMC

Perturbatively update $\latent{\rvct{z}}$, independently sample $\observed{\rvct{x}}\gvn\latent{\rvct{z}}$

<img src='images/toy-example-pm-abc-mcmc-4.svg' width='80%' /> 

Can improve efficiency for high $D\_{\latent{\rvct{z}}}$.  
However prone to sticking pathologies.

----

<!-- .slide: data-transition="none" -->
### (Pseudo-marginal) ABC MCMC

Perturbatively update $\latent{\rvct{z}}$, independently sample $\observed{\rvct{x}}\gvn\latent{\rvct{z}}$

<img src='images/toy-example-pm-abc-mcmc-5.svg' width='80%' /> 

Can improve efficiency for high $D\_{\latent{\rvct{z}}}$.  
However prone to sticking pathologies.

----

<!-- .slide: data-transition="none" -->
### (Pseudo-marginal) ABC MCMC

Perturbatively update $\latent{\rvct{z}}$, independently sample $\observed{\rvct{x}}\gvn\latent{\rvct{z}}$

<img src='images/toy-example-pm-abc-mcmc-6.svg' width='80%' /> 

Can improve efficiency for high $D\_{\latent{\rvct{z}}}$.  
However prone to sticking pathologies.

---

### ABC expectations in input space

ABC approximates expectations by introducing *kernel* e.g.

\begin{equation}
k\_{\epsilon}\lpa\,\observed{\vct{x}};\,\observed{\vct{y}}\rpa
\propto
\mathbb{I}\lsb \left|\observed{\vct{x}} - \observed{\vct{y}}| < \epsilon\right|\rsb / \epsilon^{D\_{\observed{\rvct{x}}}}
\end{equation}<!-- .element: class="fragment current-visible" data-fragment-index="1" -->

\begin{equation}
  \expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} \approx \\\\
  \frac{1}{C}
  \int\_{\latent{\set{Z}}}\int\_{\observed{\set{X}}}
    \hspace{-0.2em}
    f(\latent{\vct{z}})\,
    k\_{\epsilon}\lpa\,
      \observed{\vct{x}};\,
      \observed{\vct{y}}
    \rpa\,
    \prob{\observed{\rvct{x}},\latent{\rvct{z}}}(\dr\observed{\vct{y}},\latent{\dr\vct{z}})
\end{equation}<!-- .element: class="fragment" data-fragment-index="2" -->

ABC expectations can be rewritten as<!-- .element: class="fragment" data-fragment-index="3" -->

\begin{equation}
  \expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} \approx
  \frac{1}{C}
  \int\_{\input{\set{U}}}
    \hspace{-0.2em}
    f \circ \vctfunc{g}\_{\latent{\rvct{z}}}(\input{\vct{u}})\,
    k\_{\epsilon}\lpa\,
      \observed{\vct{x}};\,
      \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\vct{u}})
    \rpa\,
    \rho(\input{\vct{u}})\,
  \dr\input{\vct{u}}
\end{equation}<!-- .element: class="fragment" data-fragment-index="3" -->

----

### Alternative ABC MCMC operators

Construct an MCMC operator which leaves the distribution with (unnormalised) density

$$\pi\_{\epsilon}(\input{\vct{u}}) = k\_{\epsilon}\lpa\,
  \observed{\vct{x}};\,
  \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\vct{u}})
\rpa\,
\rho(\input{\vct{u}}),$$

invariant, e.g. Hamiltonian Monte Carlo, slice sampling. 

Perturbatively updating all variables in model can give large gains in high dimensions - standard MCMC approach.<!-- .element: class="fragment" data-fragment-index="1" -->


----

<!-- .slide: data-transition="none" -->
### $\epsilon \to 0$ : conditioning as a constraint

<img style='margin-bottom: -40px;' src='images/abc-in-input-space-exact-constraint-dens-v2.svg' width='80%' /> 

Exactly conditioning on observations restricts inputs to a manifold embedded in input space

$$
  \vctfunc{g}\_{\observed{\rvct{x}}}^{-1}[\observed{\vct{x}}] = \lbr \input{\vct{u}} \in \input{\set{U}} : \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\vct{u}}) = \observed{\vct{x}} \rbr.
$$

----

### Asymptotically exact inference

<div class="fragment" data-fragment-index="1">
Conditional expectations are integrals over $\vctfunc{g}\_{\observed{\rvct{x}}}^{-1}[\observed{\vct{x}}]$
  
<div style='padding-top:10px'>
\begin{equation}
  \expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} = \\\\
  \frac{1}{C}
  \int\_{\vctfunc{g}\_{\observed{\rvct{x}}}^{-1}[\observed{\vct{x}}]}
    f \circ \vctfunc{g}\_{\latent{\rvct{z}}}(\input{\vct{u}})\,
    \left| 
      \pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}
      \pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}^{\rm{T}}
    \right|^{-\frac{1}{2}}\hspace{-0.2em}
    \rho(\input{\vct{u}})\,
  \mathcal{H}^{D\_{\input{\rvct{u}}}-D\_{\observed{\rvct{x}}}}\lpa\dr\input{\vct{u}}\rpa
\end{equation}
<br />
with $\mathcal{H}^{D\_{\input{\rvct{u}}}-D\_{\observed{\rvct{x}}}}$ the Hausdorff measure on $\vctfunc{g}\_{\observed{\rvct{x}}}^{-1}[\observed{\vct{x}}]$.
</div>

<small style='font-size: 80%;'>(Diaconis, Holmes & Shahshahani; 2013)</small>
</div>

----

### Asymptotically exact inference

Sample states of a Markov chain $\lbr \input{\vct{u}^{(s)}} \rbr_{s=1}^S$ such that:

<p class="fragment" data-fragment-index="1"> the chain is restricted to $\vctfunc{g}\_{\observed{\rvct{x}}}^{-1}[\observed{\vct{x}}]$, </p>

<p class="fragment" data-fragment-index="2">
  and its stationary distribution has density wrt $\mathcal{H}^{D\_{\input{\rvct{u}}}-D\_{\observed{\rvct{x}}}}$ $\pi(\input{\vct{u}}) = \rho(\input{\vct{u}})\,
    \left| 
      \pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}} 
      \pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}\tr
    \right|^{-\frac{1}{2}} $,
</p>

<div class="fragment" data-fragment-index="3">
then we can calculate consistent estimators

$$
  \expc{\,f(\latent{\rvct{z}}) \gvn \observed{\rvct{x} = \vct{x}}} = 
  \lim\_{S \to \infty} \frac{1}{S} \sum\_{s=1}^S \lbr f \circ {\vctfunc{g}\_{\latent{\rvct{z}}}}\lpa\input{\vct{u}^{(s)}}\rpa \rbr.
$$
</div>

---

### Constrained Hamiltonian Monte Carlo <small>Hartmann and Schutte, 2005; Leli&egrave;vre, 2012; Brubaker et al. 2012</small>

Use simulated constrained Hamiltonian dynamic to propose moves on implicitly defined embedded manifold $\vctfunc{g}\_{\observed{\rvct{x}}}^{-1}[\observed{\vct{x}}]$. <!-- .element: class="fragment" data-fragment-index="1" -->

$$
  \td{\input{\vct{u}}}{t} = \vct{p}
  \qquad
  \td{\vct{p}}{t} = \pd{\log \pi}{\input{\vct{u}}} - \pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}\tr\vct{\lambda}
$$ <!-- .element: class="fragment" data-fragment-index="2" -->

subject to $\vctfunc{g}\_{\observed{\rvct{x}}}(\input{\vct{u}}) = \observed{\vct{x}}$ and $\pd{\vctfunc{g}\_{\observed{\rvct{x}}}}{\input{\vct{u}}}\vct{p} = \vct{0}$. <!-- .element: class="fragment" data-fragment-index="2" -->

<p class="fragment" data-fragment-index="3">Integrators such as RATTLE <small style='font-size: 80%;' >(Andersen, 1983)</small> time-reversible and measure preserving <small style='font-size: 80%;' >(Leimkuhler and Skeel, 1994)</small>.</p>

----

### Constrained HMC in toy example

<video autoplay controls loop>
  <source data-src="images/chmc-animation-io.mp4" type="video/mp4" />
</video>

---

### Alternative: Gaussian ABC + HMC

<div class="fragment" data-fragment-index="1" style='padding-bottom: 1em;'>
Gaussian ABC 'posterior' on $\input{\rvct{u}}$

$$\pden{\input{\rvct{u}} | \observed{\rvct{x}}}(\input{\vct{u}}\gvn\observed{\vct{x}}) \propto \exp\lbr -\frac{1}{2\epsilon^2}\left| \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\vct{u}}) - \observed{\vct{x}} \right|^2 - \log \rho(\input{\vct{u}})\rbr$$
</div>

<div class="fragment" data-fragment-index="2" style='padding-bottom: 1em';>
Augment with Gaussian distributed momenta $\rvct{p}$

$$H(\input{\vct{u}},\,\vct{p}) = \frac{1}{2\epsilon^2} \left| \vctfunc{g}\_{\observed{\rvct{x}}}(\input{\vct{u}}) - \observed{\vct{x}} \right|^2 +  \log \rho(\input{\vct{u}}) + \frac{1}{2}\vct{p}\tr\vct{p}$$
</div>

<p style='font-size: 80%;' class="fragment" data-fragment-index="3">
cf. *Pseudo-Marginal Hamiltonian Monte Carlo*, Lindsten and Doucet, 2016; *Hamiltonian ABC*, Meeds, Leenders and Welling 2015.
</p>

----

### Gaussian ABC posterior in toy example


<div style='height: 100%'>
<img src='images/gaussian-abc-with-hmc-trajectory.svg' width='50%' />
</div>

----

### Gaussian ABC + HMC in toy example

<video autoplay controls loop width="1080" height="540" >
  <source data-src="images/abc-hmc-animation-io.mp4" type="video/mp4" />
</video>

---

### Lotka-Volterra parameter inference

Model: Euler-Maruyama integration of SDEs. Infer model parameters $\latent{\rvct{z}}$ given (simulated) observed sequences $\observed{\rvct{x} = \vct{x}}$.

<img src='images/lotka-volterra-observed-state-seq.svg' width='80%' /> 

$D\_{\latent{\rvct{z}}} = 4$, $D\_{\observed{\rvct{x}}} = 100$ and $D\_{\input{\rvct{u}}} = 104$. <!-- .element: class="fragment" data-fragment-index="1" -->

----

### Lotka-Volterra - standard ABC

Full observed data: ABC reject and ABC MCMC unusable.

However using 9 dimensional summary statistics ABC MCMC (uniform ball kernel, $\epsilon = 1$) appears to converge.

<img src='images/lotka-volterra-param-posterior-hist-abc-summary-thesis.svg' width='80%' /> 

----

### Lotka-Volterra - ABC in input space

Using pseudo-marginal slice sampling <small>(Murray and Graham, 2016)</small> and HMC approaches able to tractably condition on full data.

<img src='images/lotka-volterra-param-posterior-hist-abc.svg' width='80%' /> 

----

### Lotka-Volterra - constrained HMC

Using constrained HMC able to condition on full data (and for comparison 9 dimensional summaries) with $\epsilon \to 0$.

<img src='images/lotka-volterra-param-posterior-hist-chmc.svg' width='80%' /> 

----

### Lotka-Volterra - sampling efficiency

Compare sampling efficiency using estimated effective sample sizes / computational run time. 

<img src='images/lotka-volterra-param-posterior-ess-plot.svg' width='80%' /> 

---

### Binocular pose estimation

<br />

<img src='images/binocular-pose-estimates-rmse-thesis.svg' width='100%' />


----

### Monocular pose estimation

<br />

<div class='half-column'>
<img src='images/monocular-projection-obs-1.svg' width='270px' />
</div>

<div class='half-column'>
<video autoplay loop width="270" height="270" >
  <source data-src="images/pose-hand-mono-constrained-anim-1.mp4" type="video/mp4" />
</video>
</div>

---

### Conclusions

  * Inference method for differentiable generative models.  <!-- .element: class="fragment" data-fragment-index="1" -->
  * Consider conditioning as constraint on inputs. <!-- .element: class="fragment" data-fragment-index="2" -->
  * Use of gradients allows high-dimensional inference. <!-- .element: class="fragment" data-fragment-index="3" -->
  * Asymptotically exact alternative to ABC where applicable. <!-- .element: class="fragment" data-fragment-index="4" -->

---

### Acknowledgements


<div style='display: inline-block; padding: 10px;'>
   <img src='images/amos-storkey.jpg' height='150px' style='margin: 0;' />
   <div><small>Amos Storkey</small></div>
</div>
<div style='display: inline-block; padding: 10px;'>
   <img src='images/iain-murray.jpg' height='150px' style='margin: 0;' />
   <div><small>Iain Murray</small></div>
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

Python code at <i class="fa fa-github fa-fw"></i> [git.io/dgm](http://git.io/dgm)

M. M. Graham and A. J. Storkey. Asymptotically exact inference in differentiable generative models.  
To appear in *Electronic Journal of Statistics*. Pre-print on [matt-graham.github.io](http://matt-graham.github.io)

Shorter *AISTATS* 2017 conference paper  <i class="ai ai-arxiv fa-fw"></i> [arxiv.org/abs/1605.07826](https://arxiv.org/abs/1605.07826)

---

<!-- .slide: style="font-size: 70%" -->

<h3 style='font-size: 200%;'>References</h3>
   
*  H. C. Andersen.  
   RATTLE: A 'velocity' version of the SHAKE algorithm for molecular dynamics calculations.  
   *Journal of Computational Physics.*, 1983.
   
*  M. A. Brubaker, M. Saelzmann, and R. Urtasun.  
   A family of MCMC methods on implicitly defined manifolds.  
   *AISTATS*, 2012.

*  P. Diaconis, S. Holmes and M. Shahshahani.  
   Sampling from a Manifold.  
   *Advances in Modern Statistical Theory and Applications*, 2013.

*  I. Murray and M. M. Graham.  
   Pseudo-marginal slice sampling.  
   *AISTATS*, 2016.

----

<!-- .slide: style="font-size: 70%" -->

<h3 style='font-size: 200%;'>References</h3>

*  C. Hartmann and C. Schutte.  
   A constrained hybrid Monte Carlo algorithm and the problem of calculating the free energy in several variables.  
   *ZAMM-Zeitschrift f&uuml;r Angewandte Mathematik*, 2005.
   
*  D. P. Kingma and M. Welling.  
   Auto-encoding variational Bayes.  
   *ICLR*, 2014.
   
*  B. J. Leimkuhler and R. D. Skeel.  
   Symplectic numerical integrators in constrained Hamiltonian systems.  
   *Journal of Computational Physics*, 1994.
   
*  T. Leli&egrave;vre, M. Rousset and G. Stoltz.  
   Langevin dynamics with constraints and computation of free energy differences.  
   *Mathematics of Computation*, 2012.

---

### Automatic differentiation - forward pass

Computation graph for $c = \log \mathcal{N}(x \gvn m, s)$

<img src='images/normal-log-density-computation-graph.svg'  width='80%' /> 

----

### Automatic differentiation

Backwards pass - calculate $\pd{c}{x}$, $\pd{c}{m}$ and $\pd{c}{s}$

<img src='images/normal-log-density-computation-graph-reverse-mode-ad.svg'  width='80%' /> 

---

### Constrained step

<img src='images/constrained-step-0.svg' /> 

----

### Constrained step

<img src='images/constrained-step-1.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-2.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-3.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-4.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-5.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-6.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-7.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-8.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-9.svg'  /> 

----

### Constrained step

<img src='images/constrained-step-10.svg'  /> 

---

### Structure in generator Jacobian

<div>

<div class='half-column fragment' data-fragment-index='1'>
<p>*Independent*</p>

<div class='img-row'>
<img src='images/iid-directed-generative-model.svg' height='300px' />
<img src='images/jacobian-structure-independent.svg' height='20%' />
</div>

</div>

<div class='half-column fragment' data-fragment-index='2'>
<p>*Markovian*</p>

<div class='img-row'>
<img src='images/markov-directed-generative-model.svg' height='300px' />
<img src='images/jacobian-structure-markov.svg' height='20%' />

</div>

</div>

<div style='clear: both;'></div>

</div>


---

### MNIST in-painting

CHMC samples (consecutive)
<img src='images/chmc-mnist-samples.png' width='80%' />

HMC samples (thinned by factor 40)
<img src='images/hmc-mnist-samples.png' width='80%' />

----

### MNIST in-painting

<img src='images/hmc-mnist-samples-consecutive.png' width='80%' />
