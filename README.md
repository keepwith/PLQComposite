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
Three types of input for PLQ Loss are accepted. One is the coefficients of each piece with cutoffs $\text{plq}$(default form), another is the coefficients only and takes the maximum of each piece $\text{max}$, the other is the linear version based on a series of given points $\text{points}$.  

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
\tag{max} 
$$


or 


\begin{equation}
L(z)=
\begin{cases}
\ y_1  + \frac{y_{2} - y_{1}} { x_{2} - x_{1} } (z - x_{1}), & \text{if } z \leq x_1, \\
\ y_{i-1} + \frac{y_{i} - y_{i-1}} { x_{i} - x_{i-1} } (z - x_{i-1}), & \text{if } x_{i-1} < z \leq x_{i}, i=2,...,n \\
\ y_{n-1} + \frac{y_{n-1} - y_{n}} { x_{n-1} - x_{n} } (z - x_{n}), & \text{if } z > x_{n}.
\end{cases}
\tag{points}
\end{equation}


where $\lbrace (x_1,y_1), (x_2,y_2), ..., (x_n, y_n) \rbrace$ is a series of given points and $n\geq 2$   

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
$$L_i(z_i)=c_{i}L(p_{i}z_{i}+q_{i})$$  
For Regression Problems, $L_i(z_i)=c_{i}L(y_{i}-z_{i})$.   
For Classification Problems, $L_i(z_i)=c_{i}L(y_{i}z_{i})$.  

You call **affine_transformation** method to broadcast by specifying $p_{i}$ and $q_{i}$ or just input form='regression' or 'classification'  
```python
from plqcom import affine_transformation
# specify p and q
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=y, q=0)
# form = 'classification'
rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, form='classification')
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