# <small>A brief introduction to</small> <br /> Automatic Differentiation

<img src='images/horizontal-comp-graph-neg-log-dens.svg' height='200' />

## Matt Graham <small>[&lt;matt-graham.github.io&gt;](http://matt-graham.github.io)</small>

---

## What is automatic <br />differentiation (AD)?

<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" --> An approach for computing the derivatives of a function composed of *primitive operations* with known derivatives by algorithmically applying the rules of differentiation.


<!-- .element: class="fragment" data-fragment-index="2" --> Distinct from but related to both *numerical differentiation* and *symbolic differentiation*.


----

## Advantages of AD

  * <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" --> **Efficient**: can evaluate the Jacobian of a function $f : \reals^N \to \reals^M$ at a computational cost which is $\mathcal{O}\left(\min(M, N)\right)$ times the cost of evaluating $f$.
  * <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="2" --> **Expressive**: applicable to any differentiable function which can be expressed algorithmically including use of control flow.
  * <!-- .element: class="fragment" data-fragment-index="3" --> **Exact**: provides exact derivatives of a function (modulo usual errors from using floating-point arithmetic).

---

## Numerical differentiation

Finite difference methods based on limit definitions of (partial) derivatives.
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

For example for a smooth function $f : \reals^N \to \reals^M$ if $\mathbf{e}_i$ is the length $N$ vector with 1 at the $i$th entry and zeros elsewhere then for a small positive $h$
<!-- .element: class="fragment" data-fragment-index="1" -->

$$
  \partial_i\, f(x) \approx \frac{f(x+h\mathbf{e}_i)-f(x)}{h}
$$
<!-- .element: class="fragment" data-fragment-index="1" -->

(first order forward finite difference formula).
<!-- .element: class="fragment" data-fragment-index="2" -->

----

## Numerical differentiation

To approximate the full $M\times N$ Jacobian matrix $\partial f$ need to compute finite differences for each of $\mathbf{e}_1 \dots \mathbf{e}_N$ $\implies$ cost $\mathcal{O}(N)$ evaluations of $f$.
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

This is particularly burdensome for calculating the gradient of scalar functions of a large number of variables ($M=1$, $N \gg 1$).
<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

For small $h$ can become numerically instable. 
<!-- .element: class="fragment" data-fragment-index="2" -->


----

## Symbolic differentiation

Implementations of rules of calculus in computer algebra systems (e.g. Mathematica, SymPy).
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

Gives human-readable expressions as output but verbose for complicated compositions of functions and redundancy between partial derivatives.
<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

AD instead exploits modularity of functions, computing derivatives in terms of intermediate values rather than expanding in terms of inputs.
<!-- .element: class="fragment" data-fragment-index="2" -->


---

## Jacobians

For a function $f : \reals^N \to \reals^M$ the Jacobian $\partial f(x)$ at an input $x \in \reals^N$ can be represented as a $M \times N$ matrix of partial derivatives

$$ \partial f(x) = \begin{bmatrix} 
  \partial_1{f_1}(x) & \dots & \partial_N f(x) \\\\ 
  \vdots & \ddots & \vdots \\\\
  \partial_1 f_M(x) & \dots & \partial_N f_M(x)
  \end{bmatrix}.
$$
<!-- .element: class="fragment" data-fragment-index="1" -->

More generally the Jacobian is a linear map from the domain to the codomain of a function.
<!-- .element: class="fragment" data-fragment-index="2" -->

----

## Jacobian vector products

<p class="fragment semi-fade-out" data-fragment-index="1">As the Jacobian $\partial f(x)$ is a linear map $\reals^N \to \reals^M$ we can apply it to a vector $v \in \reals^N$. We term this operation a *Jacobian vector product* (`JVP`)</p>

$$\texttt{JVP}(\,f)(x)(v) := \partial f(x) v$$
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

Cost of $\texttt{JVP}(\,f)(x)(v)$ = $\mathcal{O}(1)\times \,$ cost of  $f(x)$.
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Vector Jacobian products

<p class="fragment semi-fade-out" data-fragment-index="1">The *transpose* of the Jacobian $\partial f(x)\tr$ is a linear map $\reals^M \to \reals^N$ can be applied to a vector $v \in \reals^M$. We term this a *vector Jacobian product* (`VJP`) (cf. adjoint operator)</p>

$$\texttt{VJP}(\,f)(x)(v) := \partial f(x)\tr v = \left( v\tr \partial f(x)\right)\tr$$
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

Cost of $\texttt{VJP}(\,f)(x)(v)$ = $\mathcal{O}(1)\times \,$ cost of  $f(x)$.
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Chain rule

Chain rule for functions $f : \set{X} \to \set{Y}$ and $g : \set{Y} \to \set{Z}$
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

$$\partial (\,f\circ g) = (\partial f \circ g) \, \partial g$$
<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" -->

or equivalently if $y = g(x)$ and $z = f(y)$
<!-- .element: class="fragment" data-fragment-index="1" -->

$$\pd{z}{x} = \pd{z}{y} \pd{y}{x} = \partial f(y) \partial g(x).$$
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Forward-mode accumulation

\begin{align}
  \texttt{JVP}(\,f\circ g)(x)(v) 
  &= \partial (\,f\circ g)(x) \, v
  \\\\
  &= (\partial f \circ \underbrace{g(x)}_{y=g(x)}) \, \partial g(x) \, v
  \\\\
  &= \partial f(y) \, \partial g(x) v
  \\\\
  &=  \texttt{JVP}(\,f)(y)(\texttt{JVP}(g)(x)(v))
\end{align}

Can evaluate $g(x)$ at same time as $\texttt{JVP}(g)(x)$.
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Reverse-mode accumulation

\begin{align}
  \texttt{VJP}(\,f\circ g)(x)(v) 
  &=  \partial (\,f\circ g)(x)\tr v
  \\\\
  &= \partial g(x) \tr (\partial f \circ \underbrace{g(x)}_{y=g(x)}) \tr v
  \\\\
  &= \partial g(x) \tr \, \partial f(y) \tr v
  \\\\
  &=  \texttt{VJP}(g)(x)(\texttt{VJP}(\,f)(y)(v))
\end{align}

<p class="fragment" data-fragment-index="1">
Need to evaluate $g(x)$ *before* evaluating $\texttt{VJP}(g)(x)$.
</p>

----

## Forward- and reverse-mode AD

<p class="fragment semi-fade-out" data-fragment-index="1">
For a function which is an arbitrary composition of primitives which we can evaluate `JVP`s or
`VJP`s for, by iteratively applying the chain rule we can compute a `JVP` or `VJP` for the whole function.
</p>

<p class="fragment" data-fragment-index="1">
In the case of reverse-mode accumulation using `VJP`s we must first compute and store all intermediate values in a *forward pass* before propagating the derivatives from the outputs to inputs in a *backwards pass*.
</p>

----

## Computing gradients

For a scalar-valued function $f : \reals^N \to \reals$ we can express its gradient as a `VJP`

$$\nabla f(x) \tr = \partial f(x)\tr [1] = \texttt{VJP}(\,f)(x)([1]).$$
<!-- .element: class="fragment" data-fragment-index="1" -->

Using reverse-mode accumulation we can therefore compute $\nabla f(x)$ at similar cost to evaluating $f(x)$.
<!-- .element: class="fragment" data-fragment-index="2" -->

---

## Normal negative log density example

Consider computing the derivatives negative log density of a univariate normal distribution with mean $m$ and standard deviation $s$ at a point $x$

$$ c = \frac{1}{2}\left(\frac{x-m}{s}\right)^2 + \log (s) + \frac{1}{2}\log (2\pi) $$
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## <a href='https://live.sympy.org/?evaluate=x%2C%20m%2C%20s%20%3D%20symbols(%27x%20m%20s%27)%0Ac%20%3D%20((x-m)%2Fs)**2%20%2F%202%20%2B%20log(s)%20%2B%20log(2%20*%20pi)%20%2F%202%0Asimplify((diff(c%2C%20x)%2C%20diff(c%2C%20m)%2C%20diff(c%2C%20s)))%0A%23--%0A'>Symbolic differentiation: SymPy</a>

```Python
import sympy as sp

x, m, s = sp.symbols('x m s')
c = ((x - m) / s)**2 / 2 + sp.log(s) + sp.log(2 * sp.pi) / 2
sp.simplify((sp.diff(c, x), sp.diff(c, m), sp.diff(c, s)))
```
<!-- .element: class="fragment" data-fragment-index="1" -->

<span class="fragment" data-fragment-index="2" style='font-size: 70%'>
$$\left ( \frac{- m + x}{s^{2}}, \quad \frac{m - x}{s^{2}}, \quad \frac{s^{2} - \left(m - x\right)^{2}}{s^{3}}\right )$$
</span>

Each derivative is evaluated separately and no sharing of common subexpressions.
<!-- .element: class="fragment" data-fragment-index="3" -->

----

## Numerical differentiation: NumPy

```Python
import numpy as np

def neg_log_dens(x, m, s):
    return ((x - m) / s)**2 / 2 + np.log(s) + np.log(2 * np.pi) / 2

x, m, s, h = 0.5, 1.2, 1.1, 1e-8
c = neg_log_dens(x, m, s)
(
    (neg_log_dens(x + h, m, s) - c) / h, 
    (neg_log_dens(x, m + h, s) - c) / h, 
    (neg_log_dens(x, m, s + h) - c) / h
)

# Output: (-0.5785124, 0.5785124, 0.5409467)
```
<!-- .element: class="fragment small-code" data-fragment-index="1" -->

Four evaluations of function and approximate output.
<!-- .element: class="fragment" data-fragment-index="2" -->

----

## Automatic differentiation: Autograd


```python
import autograd.numpy as np
from autograd import grad

def neg_log_dens(x, m, s):
    return ((x - m) / s)**2 / 2 + np.log(s) + np.log(2 * np.pi) / 2
    
x, m, s = 0.5, 1.2, 1.1
grad(neg_log_dens, argnum=(0, 1, 2))(x, m, s)

# Output: (array(-0.5785124), array(0.5785124), array(0.54094666))
```
<!-- .element: class="fragment small-code" data-fragment-index="1" -->

One forward and backward pass to evaluate all 3 derivatives and 'exact' output.
<!-- .element: class="fragment" data-fragment-index="2" -->

----

## Computational graph

```python
t0 = x - m
t1 = t0 / s
t2 = np.log(s)
t3 = t1**2
t4 = t2 + 0.5 * np.log(2 * np.pi)
t5 = 0.5 * t3
c = t4 + t5
```
<!-- .element: class="fragment" data-fragment-index="1" -->

<img src='images/horizontal-comp-graph-neg-log-dens.svg' class="fragment" data-fragment-index="2" height='250' />

<!--
```graphviz

digraph G
{
    rankdir = "LR"; 
    bgcolor="transparent";
    graph [pad="0.3", ranksep="0.3", nodesep="0.3"];
    subgraph vars {
        node [color=DarkGreen fontname=Courier shape=circle width=0.6];
        x; m; s; t0; t1; t2; t3; t4; t5; c
    }
    subgraph consts {
        node [shape=plaintext fontname=Courier];
        halflog2pi[label="log(2*pi)/2"];
        two[label="2"]
    }
    subgraph ops {
        node [shape=box color="#006EAF" fontname=Courier];
        xm_t0[label="subtract"];
        t0s_t1[label="divide"];
        s_t2[label="log"];
        t1_t3[label="square"];
        t2halflog2pi_t4[label="add"];
        t3half_t5[label="divide"];
        t4t5_c[label="add"]
    }
    subgraph edges {
        {x m} -> xm_t0 -> t0;
        {t0 s} -> t0s_t1 -> t1;
        s -> s_t2 -> t2;
        t1 -> t1_t3 -> t3;
        {t2 halflog2pi} -> t2halflog2pi_t4 -> t4;
        {t3 two} -> t3half_t5 -> t5;
        {t4 t5} -> t4t5_c -> c;
    }
}
```
-->

----

## Forward-mode accumulation

<img src='images/vertical-comp-graph-neg-log-dens.svg' class="fragment" data-fragment-index="1" height='500' />
<img src='images/forward-mode-neg-log-dens.svg' class="fragment" data-fragment-index="2" height='500' />

<!--
```graphviz

digraph G
{
    rankdir = "TB"; 
    bgcolor="transparent";
    graph [pad="0.3", ranksep="0.3", nodesep="0.3"];
    subgraph vars {
        node [color=DarkGreen fontname=Courier shape=circle width=0.6];
        x[label=<<u>x</u>>]; 
        m[label=<<u>m</u>>];  
        s[label=<<u>s</u>>];  
        t0[label=<<u>t0</u>>];  
        t1[label=<<u>t1</u>>];  
        t2[label=<<u>t2</u>>];  
        t3[label=<<u>t3</u>>];  
        t4[label=<<u>t4</u>>];  
        t5[label=<<u>t5</u>>];  
        c[label=<<u>c</u>>]; 
    }

    subgraph ops {
        node [shape=box color="#006EAF" fontname=Courier];
        xm_t0[label="JVP(subtract)(x, m)"];
        t0s_t1[label="JVP(divide)(t0, s)"];
        s_t2[label="JVP(log)(s)"];
        t1_t3[label="JVP(square)(t1)"];
        t2halflog2pi_t4[label="JVP(add)(t2, log(2*pi)/2)"];
        t3half_t5[label="JVP(divide)(t3, 2)"];
        t4t5_c[label="JVP(add)(t4, t5)"]
    }
    subgraph edges {
        {x m} -> xm_t0 -> t0;
        {t0 s} -> t0s_t1 -> t1;
        s -> s_t2 -> t2;
        t1 -> t1_t3 -> t3;
        {t2} -> t2halflog2pi_t4 -> t4;
        {t3} -> t3half_t5 -> t5;
        {t4 t5} -> t4t5_c -> c;
    }
}
```
-->

<!--
```Python
dt0_dx, dt0_dm, dt0_ds = 1, -1, 0
dt1_dx, dt1_dm, dt1_ds = (1 / s) * dt0_dx, (1 / s) * dt0_dm, -t0 / s**2
dt2_dx, dt2_dm, dt2_ds = 0, 0, 1 / s
dt3_dx, dt3_dm, dt3_ds = (2 * t1) * dt1_dx, (2 * t1) * dt1_dm , (2 * t1) * dt1_ds
dt4_dx, dt4_dm, dt4_ds = dt2_dx, dt2_dm, dt2_ds
dt5_dx, dt5_dm, dt5_ds = (1 / 2) * dt3_dx, (1 / 2) * dt3_dm, (1 / 2) * dt3_ds
dc_dx, dc_dm, dc_ds = dt4_dx + dt5_dx, dt4_dm + dt5_dm, dt4_ds + dt5_ds
```


```Python
dt0_dx, dt0_dm = 1, -1
dt1_dx, dt1_dm, dt1_ds = (1 / s) * dt0_dx, (1 / s) * dt0_dm, -t0 / s**2
dt2_ds = 1 / s
dt3_dx, dt3_dm, dt3_ds = dt1_dx * (2 * t1), dt1_dm * (2 * t1), dt1_ds * (2 * t1)
dt4_ds = dt2_ds
dt5_dx, dt5_dm, dt5_ds = dt3_dx * 0.5, dt3_dm * 0.5, dt3_ds * 0.5
dc_dx, dc_dm, dc_ds = dt5_dx, dt5_dm, dt4_ds + dt5_ds
```
-->

----

## Reverse-mode accumulation

<img src='images/vertical-comp-graph-neg-log-dens.svg' class="fragment" data-fragment-index="1" height='500' />
<img src='images/reverse-mode-neg-log-dens.svg' class="fragment" data-fragment-index="2" height='500' />

<!--

```graphviz

digraph G
{
    rankdir = "BT"; 
    bgcolor="transparent";
    graph [pad="0.3", ranksep="0.3", nodesep="0.3"];
    subgraph vars {
        node [color=DarkGreen fontname=Courier shape=circle width=0.6];
        x[label=<<o>x</o>>]; 
        m[label=<<o>m</o>>];  
        s[label=<<o>s</o>>];  
        t0[label=<<o>t0</o>>];  
        t1[label=<<o>t1</o>>];  
        t2[label=<<o>t2</o>>];  
        t3[label=<<o>t3</o>>];  
        t4[label=<<o>t4</o>>];  
        t5[label=<<o>t5</o>>];  
        c[label=<<o>c</o>>]; 
    }
    subgraph ops {
        node [shape=box color="#006EAF" fontname=Courier];
        xm_t0[label="VJP(subtract)(x, m)"];
        t0s_t1[label="VJP(divide)(t0, s)"];
        s_t2[label="VJP(log)(s)"];
        t1_t3[label="VJP(square)(t1)"];
        t2_t4[label="VJP(add)(t2, log(2*pi)/2)"];
        t3_t5[label="VJP(divide)(t3, 2)"];
        t4t5_c[label="VJP(add)(t5, t4)"]
    }
    subgraph edges {
        t0 -> xm_t0 -> {x m};
        t1 -> t0s_t1 -> {t0 s};
        t2 -> s_t2 -> s;
        t3 -> t1_t3 -> t1;
        t4 -> t2_t4 -> t2;
        t5 -> t3_t5 -> t3;
        c -> t4t5_c -> {t4 t5};
    }
}
```
-->

<!--

```Python
dc_dt4, dc_dt5 = 1, 1
dc_dt3 = dc_dt5 * (1 / 2)
dc_dt2 = dc_dt4
dc_dt1 = dc_dt3 * (2 * t1)
dc_ds = dc_dt2 * (1 / s)
dc_dt0, dc_ds = dc_dt1 * (1 / s), dc_ds + dc_dt1 * (-t0 / s**2)
dc_dx, dc_dm = dc_dt0, -dc_dt0
```

```Python
dc_dc = 1
dc_dt4, dc_dt5 = dc_dc, dc_dc
dc_dt3 = dc_dt5 * 0.5
dc_dt2 = dc_dt4
dc_dt1 = dc_dt3 * (2 * t1)
dc_dt0 = dc_dt1 * (1 / s)
dc_ds = dc_dt2 * (1 / s) + dc_dt1  * (-t1 / s**2)
dc_dx = dc_dt0
dc_dm = -dc_dt0
```

-->

---

## History

AD has a long history dating back to the 1960s.

<span class="fragment" data-fragment-index="1">R.E. Wengert. *A simple automatic derivative evaluation program*.
CACM 7(8):463&ndash;4, 1964.</span>

> <!-- .element: class="fragment" data-fragment-index="2" -->
> A procedure for automatic evaluation of total  / partial derivatives of 
> arbitrary algebraic functions is presented ... 
> The key to the method is the decomposition of the given function, by 
> introduction of intermediate variables, into a series of elementary
> functional steps.


In machine learning reverse-mode AD was historically known as backpropagation.
<!-- .element: class="fragment" data-fragment-index="3" -->

----

## Computational frameworks

Increasing number of numerical computing frameworks with AD functionality, e.g.

<table class='image-table align-table fragment' data-fragment-index="2">
<tr>
<td>
<img width='200' style='padding: 10px; border: none; box-shadow: none;' src='images/theano-logo.svg' />
</td>
<td>
<img width='100' style='padding: 10px; border: none; box-shadow: none;' src='images/stan-logo.svg' /> 
</td>
<td>
<img width='150' style='padding: 10px; border: none; box-shadow: none;' src='images/tensorflow-logo.svg' />
</td>
<td>
</td>
</tr>
<tr>
<td>
<span style='font-size: 150%;'>Autograd</span>
</td>
<td>
<img width='200' style='padding: 10px; border: none; box-shadow: none;' src='images/pytorch-logo.svg' />
</td>
<td>
<img width='150' style='padding: 10px; border: none; box-shadow: none;' src='https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png' />
</td>
</tr>
</table>


---

## Autograd

Python library which can automatically differentiate native Python and NumPy code.

<!-- .element: class="fragment" data-fragment-index="1" -->Originally developed by Dougal Maclaurin as part of PhD thesis and subsequently made an open source project at https://github.com/HIPS/autograd.

----

## Autograd

Can handle arbitrary control flow and use of Python data structures such as lists and dictionaries.

<p class="fragment" data-fragment-index="1">In many cases can reuse existing NumPy based code by replacing <code class=' hljs' style='display: inline;'>import numpy as np</code> with <code class='hljs' style='display: inline;'>import autograd.numpy as np</code></p>

----

## Autodidact

A simplified Autograd implementation for pedagogical purposes by Matt Johnson

https://github.com/mattjj/autodidact

General purpose reverse-mode AD implementation in ~ 350 lines of Python.
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Autodidact

Not covered: *VSpaces* (complex support, Python containers), *forward mode*, sparse operations, SciPy functions + subset of NumPy functionality.

Example code in slides taken from Autodidact (with some minor edits for brevity and  clarity).
<!-- .element: class="fragment" data-fragment-index="1" -->


----

## Autograd - wrapped NumPy function

<img width='650' src='images/autograd-internals-original-numpy-2.svg' />


----

## Autograd - boxes and primitives

<img width='650'  src='images/autograd-internals-boxes-and-primitives.svg' />

----

## Autograd - nodes

<img width='650'  src='images/autograd-internals-nodes.svg' />

----

## Autograd - parents

<img width='650'  src='images/autograd-internals-parents.svg' />

----

## Autograd - recipes

<img width='650'  src='images/autograd-internals-recipe.svg' />

----

## Primitives

```python
def primitive(f_raw):
    def f_wrapped(*args, **kwargs):
        boxed_args, trace_id = find_top_boxed_args(args)
        if boxed_args:
            argvals = unbox(args, boxed_args)
            ans = f_wrapped(*argvals, **kwargs)
            parents = tuple(box._node for _, box in boxed_args)
            argnums = tuple(argnum for argnum, _ in boxed_args)
            recipe = (f_wrapped, ans, argvals, kwargs, argnums)
            node = Node(parents, *recipe)
            return new_box(ans, trace_id, node)
        else:
            return f_raw(*args, **kwargs)
    return f_wrapped
```
<!-- .element: class="small-code" -->


----

## Vector Jacobian products

Each differentiable primitive needs to have function(s) defining its
`VJP`(s) registered

```python
defvjp(anp.negative, lambda g, ans, x: -g)
defvjp(anp.exp, lambda g, ans, x: ans * g)
defvjp(anp.log, lambda g, ans, x: g / x)
defvjp(anp.tanh, lambda g, ans, x: g / anp.cosh(x)**2)
defvjp(anp.sinh, lambda g, ans, x: g * anp.cosh(x))
defvjp(anp.cosh, lambda g, ans, x: g * anp.sinh(x))
defvjp(anp.add, 
       lambda g, ans, x, y : unbroadcast(x, g),
       lambda g, ans, x, y : unbroadcast(y, g))
defvjp(anp.multiply, 
       lambda g, ans, x, y : unbroadcast(x, y * g),
       lambda g, ans, x, y : unbroadcast(y, x * g))
```
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Box subclasses

<!-- .element: class="fragment semi-fade-out" data-fragment-index="1" --> Key attraction of NumPy is its object-oriented design with the methods and operator overloads available for the `ndarray` class.


<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" --> Alows compact and readable syntax like  
`y = W @ x + b` rather than  
`y = np.add(np.matmul(W, x), b)`.

<!-- .element: class="fragment" data-fragment-index="2" --> As Autograd wraps arrays in boxes, it needs to also wrap the operators and methods of `ndarray` which is done via a specific `ArrayBox` subclass.

----

## Box subclasses

```Python
class ArrayBox(Box):
    @primitive
    def __getitem__(A, idx): return A[idx]
    # Constants w.r.t float data just pass though
    shape = property(lambda self: self._value.shape)
    ndim  = property(lambda self: self._value.ndim)
    size  = property(lambda self: self._value.size)
    # ...
    # Operator overloads - call Autograd wrapper functions
    def __neg__(self): return anp.negative(self)
    def __add__(self, other): return anp.add(self, other)
    def __sub__(self, other): return anp.subtract(self, other)
    def __mul__(self, other): return anp.multiply(self, other)
    # ...
```
<!-- .element: class="small-code" -->

----

## Tracing

By wrapping Python data structures in boxes and wrapping library functions as primitives Autograd can then trace the computational graphs of arbitrary compositions of the wrapped primitives.

```Python
def trace(start_node, fun, x):
    with trace_stack.new_trace() as trace_id:
        start_box = new_box(x, trace_id, start_node)
        end_box = fun(start_box)
        if (isbox(end_box) and 
               end_box._trace_id == start_box._trace_id):
            return end_box._value, end_box._node
        else:
            # Output seems independent of input
            return end_box, None
```
<!-- .element: class="fragment" data-fragment-index="1" -->


----

## Tracing

Normal negative log density example in Autograd
<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

```python
import autograd.numpy as np

def normal_neg_log_dens(x, m, s):
    return ((x - m) / s)**2 / 2 + np.log(s) + np.log(2 * pi) / 2
```
<!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->

Can be expanded as
<!-- .element: class="fragment" data-fragment-index="2" -->

```python
def normal_neg_log_dens(x, m, s):
    t0 = x - m
    t1 = t0 / s
    t2 = np.log(s)
    t3 = t1**2
    t4 = t2 + np.log(2 * pi) / 2
    t5 = t3 / 2
    return t4 + t5
```
<!-- .element: class="fragment" data-fragment-index="2" -->

----

## Tracing

Calling `trace` on this `normal_neg_log_dens` function would construct a graph of `Node` objects and `primitive` functions.

<img src='images/horizontal-comp-graph-neg-log-dens.svg' class="fragment" data-fragment-index="1" height='250' />

<!--

digraph G
{
    rankdir = "LR"; 
    bgcolor="transparent";
    graph [pad="0.3", ranksep="0.3", nodesep="0.3"];
    subgraph vars {
        node [color=DarkGreen fontname=Courier shape=circle width=0.6];
        x; m; s; t0; t1; t2; t3; t4; t5; c
    }
    subgraph consts {
        node [shape=plaintext fontname=Courier];
        halflog2pi[label="log(2*pi)/2"];
        two[label="2"]
    }
    subgraph ops {
        node [shape=box color="#006EAF" fontname=Courier];
        xm_t0[label="subtract"];
        t0s_t1[label="divide"];
        s_t2[label="log"];
        t1_t3[label="square"];
        t2halflog2pi_t4[label="add"];
        t3half_t5[label="divide"];
        t4t5_c[label="add"]
    }
    subgraph edges {
        {x m} -> xm_t0 -> t0;
        {t0 s} -> t0s_t1 -> t1;
        s -> s_t2 -> t2;
        t1 -> t1_t3 -> t3;
        {t2 halflog2pi} -> t2halflog2pi_t4 -> t4;
        {t3 two} -> t3half_t5 -> t5;
        {t4 t5} -> t4t5_c -> c;
    }
}

-->

----

## Topological sorting

Provides an ordering of the nodes in a graph so that the ancestors of a node always have a higher sort index than the node itself.

When iterating over a graph ensures child nodes are always processed before their parents.
<!-- .element: class="fragment" data-fragment-index="1" -->

<img src='images/horizontal-comp-graph-toposort-neg-log-dens.svg' class='fragment' data-fragment-index='2' height='250' />

<!--

digraph G
{
    rankdir = "LR"; 
    bgcolor="transparent";
    graph [pad="0.3", ranksep="0.3", nodesep="0.3"];
    subgraph vars {
        node [color=DarkGreen fontname=Courier shape=circle width=0.6];
        x[label=9]; 
        m[label=8]; 
        s[label=7]; 
        t0[label=6]; 
        t1[label=5]; 
        t2[label=4]; 
        t3[label=3]; 
        t4[label=2]; 
        t5[label=1]; 
        c[label=0]
    }
    subgraph consts {
        node [shape=plaintext fontname=Courier];
        halflog2pi[label="log(2*pi)/2"];
        two[label="2"]
    }
    subgraph ops {
        node [shape=box color="#006EAF" fontname=Courier];
        xm_t0[label="subtract"];
        t0s_t1[label="divide"];
        s_t2[label="log"];
        t1_t3[label="square"];
        t2halflog2pi_t4[label="add"];
        t3half_t5[label="divide"];
        t4t5_c[label="add"]
    }
    subgraph edges {
        {x m} -> xm_t0 -> t0;
        {t0 s} -> t0s_t1 -> t1;
        s -> s_t2 -> t2;
        t1 -> t1_t3 -> t3;
        {t2 halflog2pi} -> t2halflog2pi_t4 -> t4;
        {t3 two} -> t3half_t5 -> t5;
        {t4 t5} -> t4t5_c -> c;
    }
}

-->

----

## Topological sorting

```Python
def toposort(end_node):
    child_counts, stack, childless = {}, [end_node], [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts: child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)
    while childless:
        node = childless.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless.append(parent)
            else: child_counts[parent] -= 1
```
<!-- .element: class="small-code" -->

----

## Backwards pass

Given a topologically sorted computation graph can propagate derivatives backwards from outputs
to inputs by iteratively applying primitive `VJP`s

```python
def backward_pass(g, end_node):
    outgrads = {end_node: g}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        fun, value, args, kwargs, argnums = node.recipe
        for argnum, parent in zip(argnums, node.parents):
            vjp = primitive_vjps[fun][argnum]
            parent_grad = vjp(outgrad, value, *args, **kwargs)
            if parent in outgrads:
                outgrads[parent] = outgrads[parent] + parent_grad
            else:
                outgrads[parent] = parent_grad
    return outgrad
```
<!-- .element: class="fragment small-code" data-fragment-index="1" -->

----

## Reverse-mode AD - `make_vjp`<!-- .element: style="font-size: 90%" -->

The key function in the Autograd API for performing reverse-mode AD is the `make_vjp` function. 

Traces a function's forward pass and returns a function which calculates a VJP in backwards pass
<!-- .element: class="fragment" data-fragment-index="1" -->

```python
def make_vjp(fun, x):
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, fun, x)
    if end_node is None:
        def vjp(g): return np.zeros_like(x)
    else:
        def vjp(g): return backward_pass(g, end_node)
    return vjp, end_value
```
<!-- .element: class="fragment" data-fragment-index="1" -->


----

## Differential operators - `grad`<!-- .element: style="font-size: 90%" -->

Autograd also offers a series of convenience functions which work on top of `make_vjp` corresponding to various differential operators. 

<span class="fragment" data-fragment-index="1">The most commonly used is the `grad` function</span>


```Python
def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        unary_fun = lambda x: fun(
            *subval(args, argnum, x), **kwargs)
        vjp, ans = make_vjp(unary_fun, args[argnum])
        return vjp(np.ones_like(ans))
    return gradfun
```
<!-- .element: class="fragment" data-fragment-index="1" -->

----

## Differential operators - `jacobian`<!-- .element: style="font-size: 90%" -->


Autograd can also compute the $M\times N$ Jacobian of a function $\in \reals^N \to \reals^M$ by iteratively
computing VJPs for vectors $\mathbf{e}_1 ... \mathbf{e}_M$ $\implies \mathcal{O}(M)$ cost
```Python
def jacobian(fun, argnum=0):
    def jacobfun(*args, **kwargs):
        unary_fun = lambda x: fun(
            *subval(args, argnum, x), **kwargs)
        vjp, ans = make_vjp(unary_fun, args[argnum])
        jacobian_shape = ans.shape + args[argnum].shape
        basis = np.eye(ans.size).reshape(
            (ans.size,) + ans.shape)
        grads = map(vjp, basis)
        return np.reshape(np.stack(grads), jacobian_shape)
    return jacobfun
```

---

## JAX

JAX is a recently released Python library which is in some senses a successor to Autograd.

Started as a research project in Google by a group including two of the key Autograd contributors, Matt Johnson and Dougal Maclaurin. <!-- .element: class="fragment" data-fragment-index="1" -->

<!-- .element: class="fragment" data-fragment-index="2" --> Now an open source project at https://github.com/google/jax 

----

## JAX

JAX extends the tracing logic of Autograd to allow Python code to be translated to a representation (~ computational graph) that allows transformations before translating back to Python code.

<!-- .element: class="fragment" data-fragment-index="1" -->Tracing is implemented by calling functions with abstract arguments which represent the *set* of possible values.


----

## JAX

Key transformations include:

  * <!-- .element: class="fragment" data-fragment-index="1" -->reverse- and forward-mode AD via `vjp` and `jvp`  (and convenience functions such as `grad`),
  * <!-- .element: class="fragment" data-fragment-index="2" -->just-in-time compilation using XLA (backend compiler in TensorFlow) via `jit` allowing running on accelarators such as GPUs and TPUs,   
  * <!-- .element: class="fragment" data-fragment-index="3" -->automatic vectorisation / batching via `vmap`.


----

## JAX

Importantly the transformations are *composable*.

This allows for example gradient functions to be compiled to improve efficiency and efficient Jacobian calculation using vectorisation rather than sequential iteration.<!-- .element: class="fragment" data-fragment-index="1" -->

---

## References and further reading

  1. Griewank, A., 2012. Who Invented the Reverse Mode of Differentiation?. *Documenta Mathematica*, Extra Volume ISMP, pp.389-400.
  2. Baydin, A.G., Pearlmutter, B.A., Radul, A.A. and Siskind, J.M., 2018. Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18, pp.1-43.
    
