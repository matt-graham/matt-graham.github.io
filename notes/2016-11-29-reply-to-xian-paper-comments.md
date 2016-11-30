Hi Christian,

Thanks for the comments on the paper!

A few additional replies to augment what Amos wrote:

> This however sounds somewhat intense in that it involves a quasi-Newton resolution at each step. 

The method is definitely computationally expensive. If the constraint function is of the form $\boldsymbol{c} : \mathbb{R}^M \to \mathbb{R}^N$ with $M \geq N$, for large $N$ the dominant costs at each timestep are usually the constraint Jacobian $\frac{\partial \boldsymbol{c}}{\partial \boldsymbol{u}}$ evaluation (with reverse-mode automatic differentiation this can be evaluated at a cost of $\mathcal{O}(N)$ generator / constraint evaluations) and Cholesky decomposition of the Jacobian product $\frac{\partial \boldsymbol{c}}{\partial \boldsymbol{u}} \frac{\partial \boldsymbol{c}}{\partial \boldsymbol{u}}^{\rm T}$ with $\mathcal{O}(N^3)$ cost (though in many cases e.g. i.i.d. or Markovian simulated data, structure in the generator Jacobian can be exploited to give a significantly reduced cost). Each inner quasi-Newton update involves a pair of triangular solve operations which have a $\mathcal{O}(N^2)$ cost, two matrix-vector multiplications with $\mathcal{O}(MN)$ cost, and a single constraint / generator function evaluation; the number of quasi-Newton updates required for convergence in the numerical experiments tended to be much less than $N$ hence the quasi-Newton iteration tended not to be the main cost.


The high computation cost per update is traded off however with often being able to make much larger proposed moves in high-dimensional state spaces with a high chance of acceptance compared to ABC MCMC approaches. Even in the relatively small Lotka-Volterra example we provide which has an input dimension of 104 (four inputs which map to 'parameters', and 100 inputs which map to 'noise' variables), the ABC MCMC chains using the coarse ABC kernel radius $\epsilon=100$ with comparably very cheap updates were significantly less efficient in terms of effective sample size / computation time than the proposed constrained HMC approach. This was in large part due to the elliptical slice sampling updates in the ABC MCMC chains generally collapsing down to very small moves even for this relatively coarse $\epsilon$. Performance was even worse using non-adaptive ABC MCMC methods and for smaller $\epsilon$, and for higher input dimensions (e.g. using a longer sequence with correspondingly more random inputs) the comparison becomes even more favourable for the constrained HMC approach. 

A lot of the improvement seems to be coming from using gradient information: running standard HMC on the inputs $\boldsymbol{u}$ with a Gaussian ABC kernel which is equivalent to our baseline in the pose and MNIST experiments or the method proposed in the recent [*Pseudo-marginal Hamiltonian Monte Carlo*](https://arxiv.org/abs/1607.02516) by Lindsten and Doucet, seems to give comparable performance when normalised for computation time for moderate $\epsilon$. Although the standard HMC updates are much cheaper to compute, the large difference in scale between the change in density in directions normal to the (soft) constraint and in the tangent space of the constraint manifold mean a small step-size needs to be used for the standard HMC updates to maintain reasonable accept rates (with in general the manifold being non-linear and so we cannot adjust for the scale differences by a simple linear transformation / non-identity mass matrix) which means that despite the much cheaper updates standard HMC tends to give similar effective samples per computation time (and as $\epsilon$ becomes smaller this approach becomes increasingly less efficient compared to the constrained HMC method).

>  I also find it surprising that this projection step does not jeopardise the stationary distribution of the process, as the argument found therein about the approximation of the approximation is not particularly deep.

The overall simulated constrained dynamic including the projection step is symplectic on the constraint manifold (as shown in [*Symplectic numerical integrators in constrained Hamiltonian systems*](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Constraints/LeimkuhlerSymplecticConstraints94.pdf) by Leimkuhler and Reich) and time reversible, providing the projection iteration is run to convergence and an 'appropriate' step size is chosen (i.e. sufficiently small compared to the curvature of the manifold that there is guaranteed to be a solution to the projection on to the manifold and that there is only one close by solution to the projection step that will be converge to). The issues with requiring an appropriate step size to ensure a solution to the non-linear equations being solved exists and is locally unique are the same as in the implicit integrators used in Riemannian-manifold HMC methods. 

In practice we found the use of the geodesic integration scheme which splits up the unforced motion on the contraint manifold in to multiple smaller steps for each outer forced step helped in allowing an appropriate step-size for the curvature of the manifold to be chosen independently to that appropriate for the change in the density. Providing an appropriately small step size was used non-convergence was very rarely an issue and generally only in the initial updates where the geometry of the constraint manifold might be expected to be non-typical of that seen after warm-up.

> But the main thing that remains unclear to me after reading the paper is how the constraint that the pseudo-data be equal to the observable data can be turned into a closed form condition like g⁰(u)=0.

For concreteness I'll assume a parameter inference problem with parameters $\boldsymbol{\theta}$ generated from some prior distribution given a vector of standard variates $\boldsymbol{u}_1$ i.e. $\boldsymbol{\theta} = \boldsymbol{\rho}(\boldsymbol{u}_1)$ where $\boldsymbol{\rho}$ is a deterministic function.

Further let $\boldsymbol{f}(\boldsymbol{\theta},\, \boldsymbol{u}_2)$ be our simulator function which given access to a further vector of variates from a random number generator $\boldsymbol{u}_2$ and parameters $\boldsymbol{\theta} = \boldsymbol{\rho}(\boldsymbol{u}_1)$ can produce simulated data $\boldsymbol{x} = \boldsymbol{f}\left(\boldsymbol{\rho}(\boldsymbol{u}_1),\, \boldsymbol{u}_2\right)$.

If we have observed data $\boldsymbol{\breve{x}}$ then the constraint that $\boldsymbol{x} = \boldsymbol{\breve{x}}$ can be written $\boldsymbol{f}\left(\boldsymbol{\rho}(\boldsymbol{u}_1),\, \boldsymbol{u}_2\right) - \boldsymbol{\breve{x}} = \boldsymbol{0}$, so we would have $\boldsymbol{c}(\boldsymbol{u}) = \boldsymbol{f}\left(\boldsymbol{\rho}(\boldsymbol{u}_1),\, \boldsymbol{u}_2\right)- \boldsymbol{\breve{x}} = \boldsymbol{0}$ where $\boldsymbol{u} = [\boldsymbol{u}_1;\,\boldsymbol{u}_2]$.

> As mentioned above, the authors assume a generative model based on uniform (or other simple) random inputs but this representation seems impossible to achieve in reasonably complex settings.

The representation requirements can be split in to two components:

  1. We can express our simulator model in the form $\boldsymbol{y} = \boldsymbol{g}(\boldsymbol{u})$ 
  2. $\boldsymbol{g}$ is differentiable with respect to $\boldsymbol{u}$

The assumed differentiability of the generator is definitely a strong restriction, and does limit the models which this can be applied to. 

I'd argue that its quite rare that in the context of simulator type models that the first assumption isn't valid (though this may be a somewhat circular argument as it could just be what I'm defining as a simulator model is something which it applies to!). 

All (1) requires is that in the simulator code all 'randomness' is introduced by drawing random variates from a (pseudo-)random number generator where the density of the drawn variates is known (up to some potentially unknown normalisation constant). More explicitly for the parameter inference setting above if we can write our generator function in the form

```python
def generator(rng):
  params = generate_from_prior(rng)
  simulated_data = simulator(rng, params)
  return [params, simulated_data]
```

where `rng` is a pseudo-random number generator object allowing independent samples to be generated from standard densities then this assumption holds.

For (1) alone to hold the code in this function can be completely arbitrary . This could be code numerically integrating a set of stochastic differential equations, a graphics rendering pipeline or a learned parametric 'neural network' type model (with the three experiments in our paper providing toy examples of each of these).

There are degrees of freedom in what to consider the random inputs. I believe pseudo-random number generator implementations generally generate continuous random variates from just a pseudo-random uniform primitive (which itself will be generated from a pseudo-random integer primitive), e.g. RandomKit which is used for the random number implementation in (amongst other libraries) NumPy provides a `rk_double` function to generate a double-precision pseudo-random uniform variate in [0, 1) (and a `rk_gauss` function to generate a pair of Gaussian random variates but this uses `rk_double` internally). So assuming $\boldsymbol{u}$ is an arbitrarily long vector of random uniform variates will often be sufficient for allowing the simulator to be expressed as in (1) as mentioned by Dennis Prangle in the comments of your recent post 'rare events for ABC'.

In general our proposed method is actually more suited to using random inputs with unbounded support otherwise it is necessary to deal with reflections at the intersection of the constraint manifold and bounds of the support (which is possible while maintaining reversibility but a bit painful implementation wise), so for example it is better to use Gaussian variates directly than specify uniform inputs then transform to Gaussians. A uniform variate might be generated by for example setting the base density to the logistic distribution and then transforming through the logistic sigmoid. This problem of automatically transforming to unconstrained inputs has been well studied in for example Stan.

Returning to the limitations applied by assuming (2) i.e. differentiability: some of the transforms from uniform variates used to generate random variates in random number generator implementations are non-differentiable e.g. if using an accept / reject step, for example common methods for generating a Gamma variate. There are a couple of options here. We can often just use the density of the output of the transform itself e.g. a Gamma base density on the relevant $u_i$; if the parameters of the variate distribution are themselves dependent on other random inputs we need to make sure to include this dependency, but its possible to track these dependencies automatically - again probabilistic programming frameworks like Stan do just this. In other cases we might be able to use alternative (potentially less computationally efficient) transformations that avoid the accept/reject step e.g. using the original Box-Muller transform versus the more efficient but rejection based polar variant, or use tricks such as in the recent [*Rejection Sampling Variational Inference*](https://arxiv.org/pdf/1610.05683v1.pdf) by Nasesseth et al.
  
