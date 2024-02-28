# PLQ Composite Decomposition <a href="https://github.com/keepwith/PLQComposite"></a>
 
**PLQ Composite Decomposition** is designed to be a computational software package to decompose a piecewise 
linear-quadratic (PLQ) convex function as a composite ReLU-ReHU function. In this case, users can customize the
PLQ loss function and use <a href ="https://github.com/softmin/ReHLine">ReHLine</a> to solve the optimization problem.

- Github repo: [https://github.com/keepwith/PLQComposite](https://github.com/keepwith/PLQComposite)
- Documentation:[https://plqcomposite.readthedocs.io](https://plqcomposite.readthedocs.io)
- PyPI:
- Opensource License: [MIT license](https://opensource.org/licenses/MIT)

## Formulation
Given a piecewise linear-quadratic (PLQ) convex function with form

$$
f(x)=
\begin{cases}
\ a_0 x^2 + b_0 x + c_0, & \text{if } x \leq d_0, \\
\ a_i x^2 + b_i x + c_i, & \text{if } d_{i-1} < x \leq d_{i}, i=1,2,...,n \\
\ a_n x^2 + b_n x + c_n, & \text{if } x \geq d_{n}.
\end{cases}
$$

or 

$$
f(x)=max \lbrace a_0 x^2 + b_0 x + c_0, a_1 x^2 + b_1 x + c_1, ..., a_n x^2 + b_n x + c_n \rbrace.  i=1,2,...,n
$$

then this package will transform the PLQ loss function to a composite ReLU-ReHU function form below:

$$
f(x)=\sum_{l=1}^L \text{ReLU}( u_{l} x + v_{l}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{h}}( s_{h} x + t_{h}) 
$$

where $u_{l},v_{l}$ and $s_{h},t_{h},\tau_{h}$ are the ReLU-ReHU loss parameters.
The ReLU and ReHU functions are defined as $\mathrm{ReLU}(z)=\max(z,0)$ and

$$
\mathrm{ReHU}_\tau(z) =
  \begin{cases}
  \ 0,                     & z \leq 0 \\
  \ z^2/2,                 & 0 < z \leq \tau \\
  \ \tau( z - \tau/2 ),   & z > \tau
  \end{cases}.
$$

## Core Modules

### PLQLoss
A class to define a PLQ loss function. Accepted initialization can be coefficients with cutpoints, or a list of convex quadratic
functions.

### 2ReHLoss
A function to decompose a PLQ loss function as a composite ReLU-ReHU function.


```bash
pip install plqcomposite
```

## References

- [1]  Dai, B., & Qiu, Y. (2023, November). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization  with Linear 
       Computation and Linear Convergence. In *Thirty-seventh Conference on Neural Information Processing Systems*.
