# PLQ Composite Decomposition <a href="https://github.com/keepwith/PLQComposite"></a>
 
 
## Contents
- [Introduction](#Introduction)
- [Links](#Links)
- [Formulation](#Formulation)
  - [Decompose Stage](#Decompose_Stage)
  - [Broadcast Stage](#Broadcast_Stage)
- [Core Modules](#Core-Modules)
  - [PLQLoss](#PLQLoss)
    - [minimax2PLQ](#minimax2PLQ)
    - [_2ReHLoss](#_2ReHLoss)
  - [ReHProperty](#ReHProperty)
  - [Affine transformation](#Affine-transformation)
- [Examples](#Examples)
- [References](#References)


## Introduction
 

**Empirical risk minimization (ERM)** is a crucial framework that offers a general approach to handling a broad range of machine learning tasks. 

Given a general regularized ERM problem based on a convex **piecewise linear-quadratic(PLQ) loss** with the form $(1)$ below.


$$
\min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n  L_i( \mathbf{x}_{i}^\intercal \mathbf{\beta}) + \frac{1}{2} \Vert \mathbf{\beta} \Vert_2^2, \qquad \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},   \tag{1}
$$


Let $z_i=\mathbf{x}_ i^\intercal \mathbf{\beta}$, then $L_i(z_i)$ is a univariate PLQ function. 



**PLQ Composite Decomposition** is designed to be a computational software package which adopts a **two-step method** (**decompose** and **broadcast**) convert an arbitrary convex PLQ loss function in $(1)$ to a **composite ReLU-ReHU Loss** function with the form $(2)$ below. 


$$
L_i(z)=\sum_{l=1}^L \text{ReLU}( u_{l} z + v_{l}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{h}}( s_{h} z + t_{h}) \tag{2} 
$$

where $u_{l},v_{l}$ and $s_{h},t_{h},\tau_{h}$ are the ReLU-ReHU loss parameters.
The **ReLU** and **ReHU** functions are defined as 

$$\mathrm{ReLU}(z)=\max(z,0)$$ 

and


$$
\mathrm{ReHU}_\tau(z) =
  \begin{cases}
  \ 0,                     & z \leq 0 \\
  \ z^2/2,                 & 0 < z \leq \tau \\
  \ \tau( z - \tau/2 ),   & z > \tau
  \end{cases}.
$$


Finally, users can utilize <a href ="https://github.com/softmin/ReHLine">ReHLine</a> which is another useful software package to solve the ERM problem.



## Links

- Github Repo: [https://github.com/keepwith/PLQComposite](https://github.com/keepwith/PLQComposite)
- Documentation:[https://plqcomposite.readthedocs.io](https://plqcomposite.readthedocs.io)
- Open Source License: [MIT license](https://opensource.org/licenses/MIT)
- Download Repo: 
		```
		$ git clone https://github.com/keepwith/PLQComposite.git
		```


## Formulation
### Decompose Stage
In decompose stage, the main task is to convert a single convex PLQ Loss function $L(z)$
 with form $(plq)$ and $(minimax)$ to the form $(ReLU-ReHU)$

$$
L(z)=
\begin{cases}
\ a_0 z^2 + b_0 z + c_0, & \text{if } z \leq d_0, \\
\ a_i z^2 + b_i z + c_i, & \text{if } d_{i-1} < z \leq d_{i}, i=1,2,...,n-1 \\
\ a_n z^2 + b_n z + c_n, & \text{if } z > d_{n-1}.
\end{cases}
\tag{plq} 
$$


or 


$$
L(z)=max \lbrace a_{i} z^2 + b_{i} z + c_{i} \rbrace.  i=1,2,...,n
\tag{minimax} 
$$


$$
L(z)=\sum_{l=1}^L \text{ReLU}( u_{l} z + v_{l}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{h}}( s_{h} z + t_{h}) \tag{ReLU-ReHU} 
$$


### Broadcast Stage
In broadcast stage, then main task is to broadcast the $L(z)$ with the form $(ReLU-ReHU)$ in decompose stage to all the data points. i.e. generate $L_i(z_i)$ from the $L(z)$ above.

Usually, there exists a special relationship $$L_i(z_i)=c_{i}L(p_{i}z_{i}+q_{i}) \tag{b1}$$  

On the other hands, from the **Proposition 1 in [1]** the composite $(ReLU-ReHU)$ function class is closed under affine transformations.

**Proposition 1 (Closure under affine transformation).** If $L(z)$ is a composite $ReLU-ReHU$ function as in $(ReLU-ReHU)$, then for any $c>0,\  p\in\mathbb{R}, \ and \ q\in\mathbb{R}, \ cL(pz+q)$ is also composite $ReLU-ReHU$ function, that is,


$$
cL(pz+q)=\sum_{l=1}^L \text{ReLU}( u_{l}^{\prime} z + v_{l}^{\prime}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{h}^{\prime}}( s_{h}^{\prime} z + t_{h}^{\prime}), \tag{b2} 
$$


$where \ u_{l}^{\prime}=cpu_{l}, \ v_{l}^{\prime}=cu_{l}q+cv_{l}, \ \tau_{h}^{\prime}=\sqrt{c}\tau_{h},\ s_{h}^{\prime}=\sqrt{c}ps_{h}, \ and \ t_{h}^{\prime}=\sqrt{c}(s_{h}q+t_{h}).$


we combine $(b1)$ and $(b2)$, then we have

$$
L_{i}(z_{i})=c_{i}L(p_{i}z_{i}+q_{i})=\sum_{l=1}^L \text{ReLU}( u_{li}^{\prime} z_{i} + v_{li}^{\prime}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{hi}^{\prime}}( s_{hi}^{\prime} z_{i} + t_{hi}^{\prime}), \tag{b3} 
$$


substitute $(b3)$ to $(1)$ then we have


$$
\min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_ i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_ {\tau_{hi}}( s_{hi} \mathbf{x}_ i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \Vert \mathbf{\beta} \Vert_2^2, \qquad \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0}, \tag{b4}
$$


where $\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}$ and $\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}$ are the ReLU-ReHU loss parameters, and $(\mathbf{A},\mathbf{b})$ are the constraint parameters.

With the above parameters and data, we can utilize <a href ="https://github.com/softmin/ReHLine">ReHLine</a> library to solve the ERM problem.

To help you understand this operation better, we give the parameter of the broadcast of some widely used loss functions.

**Widely Used Loss Functions and Broadcast Parameters**
|  PROBLEM  | Loss($L_{i} (z_{i})$)  | $L(z)$  | Broadcast Parameters|
|  ----  | ----  | ----  | ----  |
|$SVM$ | $c_{i}(1-y_{i} z_{i})_{+}$ | $$L(z)=\begin{cases} 0 &\text{if } z < 0 \\\  z &\text{if } z \geq 0 \end{cases} $$ |$p_{i}=-y_{i}, \ q_{i}=1, \ c_{i}=c_{i}$ |
|$sSVM$ | $c_{i}ReHU_{1}(-(y_{i} z_{i}-1))$|$$L(z)=\begin{cases}\ 0 &\text{if } z < 0 \\ \ \frac{z^{2}}{2} &\text{if } 0 \leq z < 1 \\ \ z-\frac{1}{2} &\text{if } z \geq 1 \end{cases} $$ | $p_{i}=-y_{i}, \ q_{i}=1, \ c_{i}=c_{i}$ |
|$SVM^2$|$c_{i}((1-y_{i} z_{i})_{+})^{2}$ |  $$L(z)=\begin{cases}\ 0 &\text{if } z < 0 \\ \ z^{2} &\text{if } z \geq 0 \end{cases} $$|$p_{i}=-y_{i}, \ q_{i}=1, \ c_{i}=c_{i}$ |
|$LAD$|$c_{i} \| y_{i}-z_{i}\|$ | $$L(z)=\begin{cases}\ -z &\text{if } z < 0 \\ \ z &\text{if } z \geq 0 \end{cases} $$| $p_{i}=-1, \ q_{i}=y_{i}, \ c_{i}=c_{i}$ |
|$SVR$ | $c_{i} (\| y_{i}-z_{i}\|-\epsilon)_{+}$|$$L(z)=\begin{cases}\ -z-\epsilon &\text{if } z < -{\epsilon} \\ \ 0 &\text{if } -{\epsilon} \leq z < {\epsilon} \\ \ z-{\epsilon} &\text{if } z \geq {\epsilon} \end{cases} $$ | $p_{i}=-1, \ q_{i}=y_{i}, \ c_{i}=c_{i}$ |
|$QR$ | $c_{i} \rho_{\kappa}(y_{i}-z_{i})$|$$L(z)=\begin{cases}\ ({\kappa}-1)z &\text{if } z < 0 \\ \ {\kappa}z &\text{if } z \geq 0 \end{cases} $$| $p_{i}=-1, \ q_{i}=y_{i}, \ c_{i}=c_{i}$ ||






## Core Modules

### PLQLoss
A class to define a PLQ loss function. Accepted initialization can be coefficients with cutpoints, or a list of convex quadratic
functions.
#### minimax2PLQ

#### _2ReHLoss

### ReHProperty
#### Affine transformation


## Examples


## References

- [1]  Dai, B., & Qiu, Y. (2023, November). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization  with Linear Computation and Linear Convergence. In *Thirty-seventh Conference on Neural Information Processing Systems*.
- [2]  Vapnik, V. (1991). Principles of risk minimization for learning theory. In *Advances in Neural Information Processing Systems*, pages 831–838.

[Return to top](#Contents)