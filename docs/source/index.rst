.. PLQComposite Decomposition documentation master file, created by
   sphinx-quickstart on Fri Feb 23 00:06:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PLQ Composite Decomposition's documentation!
=======================================================


-  Github Repo: https://github.com/keepwith/PLQComposite
-  Documentation: https://plqcomposite.readthedocs.io
-  Open Source License: `MIT
   license <https://opensource.org/licenses/MIT>`__
-  Download Repo:
   ``$ git clone https://github.com/keepwith/PLQComposite.git``
-  Technical Details:
   `technical_details.pdf <https://github.com/keepwith/PLQComposite/blob/main/docs/technical_details.pdf>`__

Contents
--------

-  `Introduction <#Introduction>`__
-  `Usage <#Usage>`__
-  `Examples and Notebooks <#Examples-and-Notebooks>`__
-  `References <#References>`__

Introduction
------------

**Empirical risk minimization (ERM)[2]** is a crucial framework that
offers a general approach to handling a broad range of machine learning
tasks.

Given a general regularized ERM problem based on a convex **piecewise
linear-quadratic(PLQ) loss** with the form :math:`(1)` below.

.. math::


   \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n  L_i( \mathbf{x}_{i}^\intercal \mathbf{\beta}) + \frac{1}{2} \Vert \mathbf{\beta} \Vert_2^2, \qquad \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},   \tag{1}

Let :math:`z_i=\mathbf{x}_ i^\intercal \mathbf{\beta}`, then
:math:`L_i(z_i)` is a univariate PLQ function.

**PLQ Composite Decomposition** is designed to be a computational
software package which adopts a **two-step method** (**decompose** and
**broadcast**) convert an arbitrary convex PLQ loss function in
:math:`(1)` to a **composite ReLU-ReHU Loss** function with the form
:math:`(2)` below.

.. math::


   L_i(z)=\sum_{l=1}^L \text{ReLU}( u_{l} z + v_{l}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{h}}( s_{h} z + t_{h}), \tag{2}

where :math:`u_{l},v_{l}` and :math:`s_{h},t_{h},\tau_{h}` are the
ReLU-ReHU loss parameters. The **ReLU** and **ReHU** functions are
defined as

.. math:: \mathrm{ReLU}(z)=\max(z,0).

and

.. math::


   \mathrm{ReHU}_\tau(z) =
     \begin{cases}
     \ 0,                     & \text{if } z \leq 0, \\
     \ z^2/2,                 & \text{if } 0 < z \leq \tau, \\
     \ \tau( z - \tau/2 ),   & \text{if } z > \tau.
     \end{cases}

Finally, users can utilize ReHLine which is another useful software
package to solve the ERM problem.

Usage
-----

Generally Speaking, Utilize the plq composite to solve the ERM problem
need three steps below. For details of these functions you can check the
API.

1) Create a PLQ Loss and Decompose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three types of input for PLQ Loss are accepted in this package. One is
the coefficients of each piece with cutoffs (named **plq**, default
form), another is the coefficients only and takes the maximum of each
piece named **max**, the other is the linear version based on a series
of given points (named **points**). The explicit definitions of plq, max
and points are shown below.

.. math::
   \begin{equation}
   \tag{plq}
   L(z)=
   \begin{cases}
   \ a_0 z^2 + b_0 z + c_0, & \text{if } z \leq d_0, \\
   \ a_i z^2 + b_i z + c_i, & \text{if } d_{i-1} < z \leq d_{i}, \ i=1,2,...,n-1 \\
   \ a_n z^2 + b_n z + c_n, & \text{if } z > d_{n-1}.
   \end{cases}
   \end{equation}
or

.. math::

   \begin{equation}
   \tag{max}
   L(z)=max \lbrace a_{i} z^2 + b_{i} z + c_{i} \rbrace. \quad i=1,2,...,n
   \end{equation}

or

.. math::
   \begin{equation}
   \tag{points}
   L(z)=
   \begin{cases}
   \ y_1  + \frac{y_{2} - y_{1}} { x_{2} - x_{1} } (z - x_{1}), & \text{if } z \leq x_1, \\
   \ y_{i-1} + \frac{y_{i} - y_{i-1}} { x_{i} - x_{i-1} } (z - x_{i-1}), & \text{if } x_{i-1} < z \leq x_{i}, \ i=2,...,n \\
   \ y_{n-1} + \frac{y_{n-1} - y_{n}} { x_{n-1} - x_{n} } (z - x_{n}), & \text{if } z > x_{n},
   \end{cases}
   \end{equation}

where :math:`\lbrace (x_1,y_1),\ (x_2,y_2),\ ...,\ (x_n, y_n) \rbrace`
is a series of given points and :math:`n\geq 2`

**Create a PLQ Loss**

.. code:: python

   import numpy as np
   from plqcom import PLQLoss
   # plq
   plqloss1 = PLQLoss(cutpoints=np.array([0, 1, 2, 3]),quad_coef={'a': np.array([0, 0, 0, 0, 0]), 'b': np.array([0, 1, 2, 3, 4]), 'c': np.array([0, 0, -1, -3, -6])})
   # max
   plqloss2 = PLQLoss(quad_coef={'a': np.array([0., 0., 0.5]), 'b': np.array([0., -1., -1.]), 'c': np.array([0., 1., 0.5])}, form='max')
   # points
   plqloss3 = PLQLoss(points=np.array([[-3, 0], [0, 0], [1, 1], [2, 2]]), form="points")

Then call **plq_to_rehloss** method to decompose it to form :math:`(2)`

.. code:: python

   from plqcom import plq_to_rehloss
   rehloss = plq_to_rehloss(plqloss1)

2) Broadcast to all Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Usually, there exists a special relationship between each
  :math:`L_{i}`

  .. math:: L_i(z_i)=c_{i}L(p_{i}z_{i}+q_{i}).
| For Regression Problems, :math:`L_i(z_i)=c_{i}L(y_{i}-z_{i})`.
| For Classification Problems, :math:`L_i(z_i)=c_{i}L(y_{i}z_{i})`.

You call **affine_transformation** method to broadcast by specifying
:math:`p_{i}` and :math:`q_{i}` or just input form=‘regression’ or
‘classification’. You should be very careful when directly specify the
forms.

If the special relationship does not exist in your task, you can also
manually repeat stage 1 and combines all the rehloss together and then
use rehline to solve the problem.

.. code:: python

   from plqcom import affine_transformation
   # specify p and q
   rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=y, q=0)
   # form = 'classification'
   rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, form='classification')

3) Use Rehline solve the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from rehline import ReHLine
   clf = ReHLine(loss={'name': 'custom'}, C=C)
   clf.U, clf.V, clf.Tau, clf.S, clf.T= rehloss.relu_coef, rehloss.relu_intercept,rehloss.rehu_cut, rehloss.rehu_coef, rehloss.rehu_intercept
   clf.fit(X=X)
   print('sol privided by rehline: %s' % clf.coef_)

Examples and Notebooks
----------------------

-  `Hinge and Square
   loss <https://github.com/keepwith/PLQComposite/blob/main/examples/ex1_hinge_square.ipynb>`__
-  `SVM <https://github.com/keepwith/PLQComposite/blob/main/examples/ex2_svm.ipynb>`__
-  `Ridge
   Regression <https://github.com/keepwith/PLQComposite/blob/main/examples/ex3_regression.ipynb>`__

References
----------

-  [1] Dai, B., & Qiu, Y. (2023, November). ReHLine: Regularized
   Composite ReLU-ReHU Loss Minimization with Linear Computation and
   Linear Convergence. In *Thirty-seventh Conference on Neural
   Information Processing Systems*.
-  [2] Vapnik, V. (1991). Principles of risk minimization for learning
   theory. In *Advances in Neural Information Processing Systems*, pages
   831–838.

Indices and tables
==================

.. toctree::
    :maxdepth: 1

    installation
    technical details
    examples




* :ref:`genindex`
