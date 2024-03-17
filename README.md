# PLQ Composite Decomposition <a href="https://github.com/keepwith/PLQComposite"></a>
 


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
- [Examples](#Examples-and-Notebooks)
- [References](#References)


## Introduction
 

**Empirical risk minimization (ERM)[2]** is a crucial framework that offers a general approach to handling a broad range of machine learning tasks. 

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



## Usage
Generally Speaking, Utilize the plq composite to solve the ERM problem need three steps below. For details of these functions you can check the API.  

### 1) Create a PLQ Loss and Decompose  
Two types of input for PLQ Loss are accepted. One is the coefficients of each piece with cutoffs $\text{plq}$ , the other is the coefficients only and takes the maximum of each piece $\text{minimax}$.

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

**Create a PLQ Loss**  
```python
import numpy as np
from plqcom.PLQLoss import PLQLoss
plqloss = PLQLoss(quad_coef={'a': np.array([0., 0., 0.5]), 'b': np.array([0., -1., -1.]), 'c': np.array([0., 1., 0.5])}, form='minimax')
```

Then call **plq_to_rehloss** method to decompose it to form $(2)$  
```python
from plqcom.PLQProperty import plq_to_rehloss
rehloss = plq_to_rehloss(plqloss)
```

### 2) Broadcast to all Samples
Usually, there exists a special relationship between each $L_{i}$
$$L_i(z_i)=c_{i}L(p_{i}z_{i}+q_{i})$$  
Call **affine_transformation** method to broadcast
```python
from plqcom.ReHProperty import affine_transformation
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=y, q=0)
```
### 3) Use Rehline solve the problem
```
from rehline import ReHLine
clf = ReHLine(loss={'name': 'custom'}, C=C)
clf.U, clf.V, clf.Tau, clf.S, clf.T= rehloss.relu_coef, rehloss.relu_intercept,rehloss.rehu_cut, rehloss.rehu_coef, rehloss.rehu_intercept
clf.fit(X=X)
print('sol privided by rehline: %s' % clf.coef_)
```




## Examples and Notebooks
- [Hinge and Square loss](https://github.com/keepwith/PLQComposite/blob/main/examples/ex1_hinge_square.ipynb)
- [SVM](https://github.com/keepwith/PLQComposite/blob/main/examples/ex2_svm.ipynb)
- [Ridge Regression](https://github.com/keepwith/PLQComposite/blob/main/examples/ex3_regression.ipynb)



## References

- [1]  Dai, B., & Qiu, Y. (2023, November). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization  with Linear Computation and Linear Convergence. In *Thirty-seventh Conference on Neural Information Processing Systems*.
- [2]  Vapnik, V. (1991). Principles of risk minimization for learning theory. In *Advances in Neural Information Processing Systems*, pages 831–838.

[Return to top](#Contents)