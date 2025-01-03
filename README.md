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
 

**Empirical Risk Minimization (ERM)**[3] is a fundamental framework that provides a general methodology for addressing a wide variety of machine learning tasks. In many machine learning ERM problems, loss functions can be represented as **piecewise linear-quadratic (PLQ)** functions. Specifically, the formulation given a PLQ loss function $L_i(\cdot): \mathbb{R} \rightarrow \mathbb{R}^{+}_{0}$ is as follows:

$$
\begin{aligned}
\min_{\boldsymbol{\beta} \in \mathbb{R}^d} \sum_{i=1}^n  L_i( \mathbf{x}_{i}^\intercal \boldsymbol{\beta}) + \frac{1}{2} \Vert \boldsymbol{\beta} \Vert_2^2, \qquad \text{ s.t. } \mathbf{A} \boldsymbol{\beta} + \mathbf{b} \geq \mathbf{0},   
\end{aligned}
\tag{1}
$$


where $\mathbf{x}_{i} \in \mathbb{R}^d$ is the feature vector for the $i$-th observation, and $\boldsymbol{\beta} \in \mathbb{R}^d$ is an unknown coefficient vector. 


Our objective is to transform the form of the PLQ loss function $L_i(\cdot)$ in $(1)$ into the sum of a finite number of **rectified linear units (ReLU)** [2] and **rectified Huber units (ReHU)** [1] as follows. 


$$
\begin{aligned}
L_i(z)=\sum_{l=1}^L \text{ReLU}( u_{li} z + v_{li}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{hi}}( s_{hi} z + t_{hi}), 
\end{aligned}
\tag{2} 
$$

where $u_{li},v_{li}$ and $s_{hi},t_{hi},\tau_{hi}$ are the ReLU-ReHU loss parameters for $L(\cdot)$, and the ReLU and ReHU functions are defined as

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
In general, to solve the ERM problem using **plqcom** and **reline**, follow the four steps outlined below. For specific details about these functions, please refer to the API documentation.

### 1) Representation of PLQ functions
We consider three distinct representations of the PLQ functions, which are enumerated as follows. 

**plq**: specifying the coefficients of each piece with cutoffs.


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


**max**: specifying the coefficients of a series of quadratic functions and taking the pointwise maximum of each function.


$$
\begin{aligned}
L(z)=\max_{j=1,2,...,m} \lbrace a_{j} z^2 + b_{j} z + c_{j} \rbrace. \qquad
\end{aligned}
\tag{max} 
$$


**points**: constructing piecewise linear functions based on a series of given points.


$$
\begin{aligned}
L(z)=
\begin{cases}
\ q_1  + \frac{q_{2} - q_{1}} { p_{2} - p_{1} } (z - p_{1}), & \text{if } z \leq p_1, \\
\ q_{j-1} + \frac{q_{j} - q_{j-1}} { p_{j} - p_{j-1} } (z - p_{j-1}), \ & \text{if } p_{j-1} < z \leq p_{j}, \ j=2,...,m, \\
\ q_{m-1} + \frac{q_{m-1} - q_{m}} { p_{m-1} - p_{m} } (z - p_{m}), & \text{if } z > p_{m},
\end{cases}
\end{aligned}
\tag{points}
$$


where $\lbrace (p_1,q_1),\ (p_2,q_2),\ ...,\ (p_m, q_m) \rbrace$ are a series of given points and $m\geq 2$. The **points** representation can only express piecewise linear functions.

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

### 2) Decompose to ReLU-ReHU representation
We can call **plq_to_rehloss** method to decompose it to form $(2)$.  
```python
from plqcom import plq_to_rehloss
rehloss = plq_to_rehloss(plqloss1)
```

### 3) Affine casting
Note that, in practice, $L_i(\cdot)$ in $(1)$ can typically be obtained through affine transformation of a single *prototype loss* $L(\cdot)$, that is,


$$
  L_i(z) = C_i L(p_i z + q_i),
$$


where $C_i>0$ is the sample weight for the $i$-th instance, and $p_i$ and $q_i$ are constants. For example,

- for classification problems:


  $$
  L_i( \mathbf{x}_{i}^\intercal \boldsymbol{\beta} ) = C_{i}L(y_i \mathbf{x}_{i}^\intercal \boldsymbol{\beta});
  $$
  
  
- for regression problems:


  $$
  L_i( \mathbf{x}_{i}^\intercal \boldsymbol{\beta} ) = C_{i}L(y_i - \mathbf{x}_{i}^\intercal \boldsymbol{\beta}).
  $$
  
  


Utilize the **affine_transformation** method to broadcast by providing $p_i$ and $q_i$, or by simply indicating the input form as 'regression' or 'classification'. You should be careful when directly specifying these forms.

If the specific relationship does not apply to your task, you may manually repeat stage 1 and 2. Then, combine all the rehloss together and use **rehline** to address the problem.  

```python
from plqcom import affine_transformation
# specify p and q
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=y, q=0)
# form = 'classification'
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, form='classification')
```

### 4) Use Rehline solve the problem
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

- [1]  Dai B, Qiu Y (2024). ReHLine: regularized composite ReLU-ReHU loss minimization with linear computation and linear convergence. *Advances in Neural Information Processing Systems (NIPS)*, 36.
- [2] Fukushima K (1969). Visual feature extraction by a multilayered network of analog threshold elements. *IEEE Transactions on Systems Science and Cybernetics*, 5(4): 322–333.
- [3]  Vapnik, V. (1991). Principles of risk minimization for learning theory. In *Advances in Neural Information Processing Systems*, pages 831–838.


[Return to top](#PLQ-Composite-Decomposition)