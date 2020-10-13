# Computational mathematics for learning and data analysis [2019-2020]: project non-ML n°12 implementation
![Most significant result](/results/big_m.png)
## The problem
###(P) 
is the linear least squares problem
min<sub>w</sub>∥X<sup>+</sup>w−y∥
where X<sup>+</sup> is the matrix obtained by augmenting the (tall thin) matrix X from the ML-cup dataset by prof. Micheli with a few functions of your choice of the features of the dataset, and y is one of the corresponding output vectors. For instance, if X contains columns [x1,x2], you may add functions such as log(x1), x1<sup>2</sup>, x1*x2, …

###(A1)
is an algorithm of the class of Conjugate Gradient methods [references: *J. Nocedal, S. Wright, Numerical Optimization*].

###(A2) 
is thin QR factorization with Householder reflectors [*Trefethen, Bau, Numerical Linear Algebra, Lecture 10*], in the variant where one does not form the matrix Q, but stores the Householder vectors uk and uses them to perform (implicitly) products with Q and Q<sup>T</sup>.

No off-the-shelf solvers allowed. In particular you must implement yourself the thin QR factorization, and the computational cost of your implementation should scale linearly with the largest dimension of the matrix X.

## How to tun the code
To install the requirements with 

> pip install -r requirements.txt

To see the results of **(A1)** applied to **(P)**, execute the [Conjugate Gradient method](conjugate_gradient.py) with 

> python [test_cg.py](test_cg.py)

and for **(A2)** execute [QR factorization method](qr_method.py) with 

>python [test_qr.py](test_qr.pyt)

To test the computational cost of **(A1)** and **(A2)** applied to 3 different matrices 

* m=n, 
* m>n
* m>>n
   
use 
>python [times.py](times.py)

this will **override** the plots 

* square.png 
* little_m.png 
* big_m.png

in the results folder.
The numerical results of times.py will be saved in the txt files **{method}_{type of matrix}.txt** (for example [cg_little_m.txt](results/cg_little_m.txt) is the Conjugate Gradient applied to the matrix m>n). The format is the following: 

1. the first line are the sizes
2. the second line are the times

and everything is saved in scientific notation.