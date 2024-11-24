# PLQ Composite Decomposition<a href="https://github.com/keepwith/PLQComposite"></a>
 


- Github Repo: [https://github.com/keepwith/PLQComposite](https://github.com/keepwith/PLQComposite)
- Documentation: [https://plqcomposite.readthedocs.io](https://plqcomposite.readthedocs.io)
- Open Source License: [MIT license](https://opensource.org/licenses/MIT)
- Download Repo: 
		```
		$ git clone https://github.com/keepwith/PLQComposite.git
		```
- Technical Details: [technical_details.pdf](https://github.com/keepwith/PLQComposite/blob/main/docs/technical_details.pdf)   


## Contents
- [Introduction](#Introduction)
- [Usage](#Usage)
- [Examples and Notebooks](#Examples-and-Notebooks)
- [References](#References)


## Introduction
 

**Empirical risk minimization (ERM)[2]** is a crucial framework that offers a general approach to handling a broad range of machine learning tasks. 

Given a general regularized ERM problem based on a convex **piecewise linear-quadratic(PLQ) loss** with the form $(1)$ below.


$$
\begin{aligned}
\min_{\boldsymbol{\beta} \in \mathbb{R}^d} \sum_{i=1}^n  L_i( \mathbf{x}_{i}^\intercal \boldsymbol{\beta}) + \frac{1}{2} \Vert \boldsymbol{\beta} \Vert_2^2, \qquad \text{ s.t. } \mathbf{A} \boldsymbol{\beta} + \mathbf{b} \geq \mathbf{0},   
\end{aligned}
\tag{1}
$$


Let $z_i=\mathbf{x}_ i^\intercal \boldsymbol{\beta}$, then $L_i(z_i)$ is a univariate PLQ function. 



**PLQ Composite Decomposition** is designed to be a computational software package which adopts a **two-step method** (**decompose** and **broadcast**) convert an arbitrary convex PLQ loss function in $(1)$ to a **composite ReLU-ReHU Loss** function with the form $(2)$ below. 


$$
\begin{aligned}
L_i(z)=\sum_{l=1}^L \text{ReLU}( u_{li} z + v_{li}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{hi}}( s_{hi} z + t_{hi}), 
\end{aligned}
\tag{2} 
$$

where $u_{l},v_{l}$ and $s_{h},t_{h},\tau_{h}$ are the ReLU-ReHU loss parameters.
The **ReLU** and **ReHU** functions are defined as 

$$\mathrm{ReLU}(z)=\max(z,0).$$ 

and


$$
\mathrm{ReHU}_\tau(z) =
  \begin{cases}
  \ 0,                     & \text{if } z \leq 0, \\
  \ z^2/2,                 & \text{if } 0 < z \leq \tau, \\
  \ \tau( z - \tau/2 ),   & \text{if } z > \tau.
  \end{cases}
$$


Finally, users can utilize <a href ="https://github.com/softmin/ReHLine">ReHLine</a> which is another useful software package to solve the ERM problem.  



## Usage
Generally Speaking, Utilize the plq composite to solve the ERM problem need three steps below. For details of these functions you can check the API.  

### 1) Create a PLQ Loss and Decompose  
Three types of input for PLQ Loss are accepted in this package. One is the coefficients of each piece with cutoffs (named **plq**, default form), another is the coefficients only and takes the maximum of each piece named **max**, the other is the linear version based on a series of given points (named **points**). The explicit definitions of plq, max and points are shown below.  

$$
\begin{aligned}
L(z)=
\begin{cases}
\ a_1 z^2 + b_1 z + c_1, & \text{if } z \leq d_1, \\
\ a_j z^2 + b_j z + c_j, & \text{if } d_{j-1} < z \leq d_{j}, \ j=2,3,...,m-1 \\
\ a_m z^2 + b_m z + c_m, & \text{if } z > d_{m-1}.
\end{cases}
\end{aligned}
\tag{plq} 
$$


or 


$$
\begin{aligned}
L(z)=\max_{j=1,2,...,m} \lbrace a_{j} z^2 + b_{j} z + c_{j} \rbrace. \qquad i=1,2,...,m
\end{aligned}
\tag{max} 
$$


or 


$$
\begin{aligned}
L(z)=
\begin{cases}
\ q_1  + \frac{q_{2} - q_{1}} { p_{2} - p_{1} } (z - p_{1}), & \text{if } z \leq p_1, \\
\ q_{j-1} + \frac{q_{j} - q_{j-1}} { p_{j} - p_{j-1} } (z - p_{j-1}), \ & \text{if } p_{j-1} < z \leq p_{j}, \ j=2,...,m \\
\ q_{m-1} + \frac{q_{m-1} - q_{m}} { p_{m-1} - p_{m} } (z - p_{m}), & \text{if } z > p_{m},
\end{cases}
\end{aligned}
\tag{points}
$$


where $\lbrace (p_1,q_1),\ (p_2,q_2),\ ...,\ (p_m, q_m) \rbrace$ are a series of given points and $m\geq 2$   
The **points** representation can only express piecewise linear functions.

**Create a PLQ Loss**  
```python
import numpy as np
from plqcom import PLQLoss
# plq
plqloss1 = PLQLoss(cutpoints=np.array([0, 1, 2, 3]),quad_coef={'a': np.array([0, 0, 0, 0, 0]), 'b': np.array([0, 1, 2, 3, 4]), 'c': np.array([0, 0, -1, -3, -6])})
# max
plqloss2 = PLQLoss(quad_coef={'a': np.array([0., 0., 0.5]), 'b': np.array([0., -1., -1.]), 'c': np.array([0., 1., 0.5])}, form='max')
# points
plqloss3 = PLQLoss(points=np.array([[-3, 0], [0, 0], [1, 1], [2, 2]]), form="points")
```

Then call **plq_to_rehloss** method to decompose it to form $(2)$  
```python
from plqcom import plq_to_rehloss
rehloss = plq_to_rehloss(plqloss1)
```

### 2) Broadcast to all Samples
Usually, there exists a special relationship between each $L_{i}$
$$L_i(z_i)=c_{i}L(p_{i}z_{i}+q_{i}).$$  
For Regression Problems, $L_i(z_i)=c_{i}L(y_{i}-z_{i})$.   
For Classification Problems, $L_i(z_i)=c_{i}L(y_{i}z_{i})$.  

You call **affine_transformation** method to broadcast by specifying $p_{i}$ and $q_{i}$ or just input form='regression' or 'classification'. You should be very careful when directly specify the forms. 

If the special relationship does not exist in your task, you can also manually repeat stage 1 and combines all the rehloss together and then use rehline to solve the problem.  

```python
from plqcom import affine_transformation
# specify p and q
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=y, q=0)
# form = 'classification'
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, form='classification')
```

### 3) Use Rehline solve the problem
``` python
from rehline import ReHLine
clf = ReHLine(loss={'name': 'custom'}, C=C)
clf.U, clf.V, clf.Tau, clf.S, clf.T= rehloss.relu_coef, rehloss.relu_intercept,rehloss.rehu_cut, rehloss.rehu_coef, rehloss.rehu_intercept
clf.fit(X=X)
print('sol privided by rehline: %s' % clf.coef_)
```




## Examples and Notebooks
- [Hinge and Square loss](https://colab.research.google.com/drive/1VKsSci1DqkHt7wJgruYRN3dp1EHO87SU?usp=sharing)
- [Portfilio Optimization](https://colab.research.google.com/drive/1k2ZVk9FmtnPklA1MQpQg2-JqDbwR9RHu?usp=sharing)
- [SVM](https://github.com/keepwith/PLQComposite/blob/main/examples/ex2_svm.ipynb)
- [Ridge Regression](https://github.com/keepwith/PLQComposite/blob/main/examples/ex3_regression.ipynb)



## References

- [1]  Dai, B., & Qiu, Y. (2023, November). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization  with Linear Computation and Linear Convergence. In *Thirty-seventh Conference on Neural Information Processing Systems*.
- [2]  Vapnik, V. (1991). Principles of risk minimization for learning theory. In *Advances in Neural Information Processing Systems*, pages 831–838.

[Return to top](#PLQ-Composite-Decomposition)