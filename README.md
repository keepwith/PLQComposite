# PLQ Composite Decomposition

[![DOI](https://img.shields.io/badge/DOI-10.6339%2F24--JDS1162-blue)](https://doi.org/10.6339/24-JDS1162)

- **Paper**: [ReLU-ReHU Representations of Piecewise Linear-Quadratic Losses](https://jds-online.org/journal/JDS/article/1401/info) (*Journal of Data Science*, Vol. 23, No. 4, pp. 648–658, 2025)
- Github Repo: [https://github.com/keepwith/PLQComposite](https://github.com/keepwith/PLQComposite)
- Documentation: [https://plqcomposite.readthedocs.io](https://plqcomposite.readthedocs.io)
- Open Source License: [MIT license](https://opensource.org/licenses/MIT)
- Download Repo: 
		```
		$ git clone https://github.com/keepwith/PLQComposite.git
		```


## Contents
- [Introduction](#Introduction)
- [Usage](#Usage)
- [Examples and Notebooks](#Examples-and-Notebooks)
- [Citation](#Citation)
- [References](#References)


## Introduction
 

**Empirical Risk Minimization (ERM)**[3] is a fundamental framework that provides a general methodology for addressing a wide variety of machine learning tasks. In many machine learning ERM problems, loss functions can be represented as **piecewise linear-quadratic (PLQ)** functions. Specifically, the formulation given a PLQ loss function $L_i(\cdot): \mathbb{R} \rightarrow \mathbb{R}^{+}_{0}$ is as follows:

$$
\min_{\boldsymbol{\beta} \in \mathbb{R}^d} \sum_{i=1}^n L_i\left(\mathbf{x}_{i}^{\intercal}\boldsymbol{\beta}\right) + \frac{1}{2}\Vert\boldsymbol{\beta}\Vert_2^2, \qquad \text{s.t.}\quad \mathbf{A}\boldsymbol{\beta}+\mathbf{b}\geq\mathbf{0}.
$$


where $\mathbf{x}_{i} \in \mathbb{R}^d$ is the feature vector for the $i$-th observation, $\boldsymbol{\beta} \in \mathbb{R}^d$ is an unknown coefficient vector, and $\mathbf{A} \in \mathbb{R}^{K \times d}$, $\mathbf{b} \in \mathbb{R}^{K}$ define optional linear inequality constraints (e.g. portfolio optimization; see `ex4_portfolio.ipynb`). 


Our objective is to transform the form of the PLQ loss function $L_i(\cdot)$ in $(1)$ into the sum of a finite number of **rectified linear units (ReLU)** [2] and **rectified Huber units (ReHU)** [1] as follows. 


$$
\begin{aligned}
L_i(z)=\sum_{l=1}^L \text{ReLU}( u_{li} z + v_{li}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{hi}}( s_{hi} z + t_{hi}), 
\end{aligned}
$$

where $u_{li},v_{li}$ and $s_{hi},t_{hi},\tau_{hi}$ are the ReLU-ReHU loss parameters for $L_i(\cdot)$, and the ReLU and ReHU functions are defined as

$$\mathrm{ReLU}(z) = \max(z,0).$$ 


and


$$
\mathrm{ReHU}_\tau(z) =
  \begin{cases}
  0, & \text{if } z \leq 0, \\
  z^2/2, & \text{if } 0 < z \leq \tau, \\
  \tau( z - \tau/2 ), & \text{if } z > \tau.
  \end{cases}
$$


Finally, users can utilize <a href ="https://github.com/softmin/ReHLine-python">ReHLine</a> which is another useful software package to solve the ERM problem.  



## Usage
In general, to solve the ERM problem using **plqcom** and **reline**, follow the four steps outlined below. For specific details about these functions, please refer to the API documentation.

### 1) Representation of PLQ functions
We consider three distinct representations of the PLQ functions, which are enumerated as follows. 

**plq**: specifying the coefficients of each piece with cutoffs.


$$
\begin{aligned}
L(z)=
\begin{cases}
a_1 z^2 + b_1 z + c_1, & \text{if } z \leq d_1, \\
\qquad \cdots \\
a_j z^2 + b_j z + c_j, & \text{if } d_{j-1} < z \leq d_{j}, \ j=2,...,m-1, \\
\qquad \cdots \\
a_m z^2 + b_m z + c_m, & \text{if } z > d_{m-1}.
\end{cases}
\end{aligned}
$$


**max**: specifying the coefficients of a series of quadratic functions and taking the pointwise maximum of each function.


$$
\begin{aligned}
L(z)= \max_{j=1,2,...m} \left( a_{j} z^2 + b_{j} z + c_{j} \right).
\end{aligned}
$$


**points**: constructing piecewise linear functions based on a series of given points.


$$
\begin{aligned}
L(z)=
\begin{cases}
q_1 + \frac{q_{2} - q_{1}}{p_{2} - p_{1}} (z - p_{1}), & \text{if } z \leq p_1, \\
\qquad \cdots \\
q_{j-1} + \frac{q_{j} - q_{j-1}}{p_{j} - p_{j-1}} (z - p_{j-1}), & \text{if } p_{j-1} < z \leq p_{j}, \ j=2,...,m, \\
\qquad \cdots \\
q_{m} + \frac{q_{m} - q_{m-1}}{p_{m} - p_{m-1}} (z - p_{m}), & \text{if } z > p_{m},
\end{cases}
\end{aligned}
$$


where $\lbrace (p_1,q_1), (p_2,q_2), ..., (p_m, q_m) \rbrace$ are a series of given points and $m\geq 2$. The **points** representation can only express piecewise linear functions.

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
  L_{i} ( \mathbf{x}_{i}^{\intercal} \boldsymbol{\beta} ) = C_{i} L(y_i \mathbf{x}_{i}^{\intercal} \boldsymbol{\beta});
$$
  
  
- for regression problems:


$$
  L_{i} ( \mathbf{x}_{i}^{\intercal} \boldsymbol{\beta} ) = C_{i} L(y_i - \mathbf{x}_{i}^{\intercal} \boldsymbol{\beta}).
$$
  
  


Utilize the **affine_transformation** method to broadcast by providing $p_i$ and $q_i$, or by simply indicating the input form as 'regression' or 'classification'. You should be careful when directly specifying these forms.

**Do not confuse `c` and `C` in code** (they are related to, but not the same as, the mathematical $C_i$ above):

| Symbol | Where | Role |
|--------|-------|------|
| `c` in `affine_transformation(..., c=...)` | plqcom | Per-sample scale on the **prototype** loss after decomposition. Use `c=1` for uniform weighting. Use `c != 1` only for heterogeneous sample weights $C_i$. |
| `C` in `ReHLine(C=...)` / `plq_Ridge_*(C=...)` | rehline | Global ERM weight / inverse regularization strength — the main tuning knob for loss vs ridge penalty. |

For **rehline** $\geq$ 0.1.0, set ERM strength via `ReHLine(C=...)` (or sklearn `C=...`) only; use `c=1` in `affine_transformation` unless you need per-sample weights. **Do not** pass `c=C` when you also set `ReHLine(C=C)` — ReHLine applies `C` internally and the penalty would be doubled.

If the specific relationship does not apply to your task, you may manually repeat stage 1 and 2. Then, combine all the rehloss together and use **rehline** to address the problem.  

```python
from plqcom import affine_transformation
# specify p and q (c=1; C is set on ReHLine below)
rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, p=y, q=0)
# form = 'classification'
rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, form='classification', y=y)
```

### 4) Solve with ReHLine

ReHLine $\geq$ 0.1.0 supports two calling styles:

#### 4a) Low-level API (after plqcom decomposition)

Use this when the loss is a **custom** PLQ composite from plqcom (steps 1–3). Pass the decomposed parameters to `ReHLine` and call `fit(X)`:

```python
# C: ReHLine ERM weight (ex1/ex2 use 0.5; ex3 uses 1.0; ex4 uses 0.5)
C = 0.5
from rehline import ReHLine
clf = ReHLine(C=C)
clf._U, clf._V, clf._Tau, clf._S, clf._T = (
    rehloss.relu_coef, rehloss.relu_intercept,
    rehloss.rehu_cut, rehloss.rehu_coef, rehloss.rehu_intercept,
)
clf.fit(X=X)
print('sol provided by rehline: %s' % clf.coef_)
```

For problems with linear constraints (e.g. portfolio), also set `clf._A` and `clf._b`.

#### 4b) Scikit-learn style API (built-in losses)

For **standard** PLQ losses already built into ReHLine (SVM, hinge, Huber, MSE, MAE, ...), use `plq_Ridge_Classifier` or `plq_Ridge_Regressor` with `fit(X, y)` — no plqcom decomposition needed:

```python
# C: inverse regularization strength in ReHLine (tune per problem; see notebooks)
C = 1.0

# Classification (e.g. SVM; ex2_svm.ipynb uses C=0.5)
from rehline import plq_Ridge_Classifier
clf = plq_Ridge_Classifier(loss={'name': 'svm'}, C=C)
clf.fit(X, y)

# Regression (e.g. ridge / MSE; ex3_regression.ipynb uses C=1)
from rehline import plq_Ridge_Regressor
clf = plq_Ridge_Regressor(loss={'name': 'MSE'}, C=C)
clf.fit(X, y)

print(clf.coef_, clf.intercept_)
```

See the [ReHLine-python](https://github.com/softmin/ReHLine-python) documentation for the full list of built-in loss names.




## Examples and Notebooks

| Notebook | Description |
|----------|-------------|
| [ex1: Hinge–Square](https://colab.research.google.com/drive/1VKsSci1DqkHt7wJgruYRN3dp1EHO87SU?usp=sharing) | Custom composite classification loss; low-level `ReHLine` API only |
| [ex2: SVM](https://github.com/keepwith/PLQComposite/blob/main/examples/ex2_svm.ipynb) | Hinge SVM via plqcom decomposition **and** `plq_Ridge_Classifier` |
| [ex3: Ridge Regression](https://github.com/keepwith/PLQComposite/blob/main/examples/ex3_regression.ipynb) | MSE / ridge via plqcom **and** `plq_Ridge_Regressor` |
| [ex4: Portfolio](https://colab.research.google.com/drive/1k2ZVk9FmtnPklA1MQpQg2-JqDbwR9RHu?usp=sharing) | PLQ from points + linear constraints (`_A`, `_b`) |

Source notebooks live in `examples/` and are mirrored under `docs/source/notebooks/` for Sphinx.



## Citation

If you use **plqcom** in your research, please cite our paper:

```bibtex
@article{GaoDaiQiu2025,
  title   = {ReLU-ReHU Representations of Piecewise Linear-Quadratic Losses},
  author  = {Gao, Tingxian and Dai, Ben and Qiu, Yixuan},
  journal = {Journal of Data Science},
  volume  = {23},
  number  = {4},
  pages   = {648--658},
  year    = {2025},
  doi     = {10.6339/24-JDS1162},
  url     = {https://jds-online.org/journal/JDS/article/1401/info}
}
```

## References

- [1]  Dai B, Qiu Y (2024). ReHLine: regularized composite ReLU-ReHU loss minimization with linear computation and linear convergence. *Advances in Neural Information Processing Systems (NIPS)*, 36.
- [2] Fukushima K (1969). Visual feature extraction by a multilayered network of analog threshold elements. *IEEE Transactions on Systems Science and Cybernetics*, 5(4): 322–333.
- [3]  Vapnik, V. (1991). Principles of risk minimization for learning theory. In *Advances in Neural Information Processing Systems*, pages 831–838.


[Return to top](#PLQ-Composite-Decomposition)