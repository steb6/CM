# CMLDA
Implementation of the project for the course Computational Mathematics For Learning And Data Analysis, project numer 12 (LS with QR by Householder and Conjugate Gradient Method)

# Project 12
(P) is the linear least squares problem
minw∥X^w−y∥
where X^ is the matrix obtained by augmenting the (tall thin) matrix X from the ML-cup dataset by prof. Micheli with a few functions of your choice of the features of the dataset, and y is one of the corresponding output vectors. For instance, if X contains columns [x1,x2], you may add functions such as log(x1), x1.^2, x1.*x2, …

(A1) is an algorithm of the class of Conjugate Gradient methods [references: J. Nocedal, S. Wright, Numerical Optimization].

(A2) is thin QR factorization with Householder reflectors [Trefethen, Bau, Numerical Linear Algebra, Lecture 10], in the variant where one does not form the matrix Q, but stores the Householder vectors uk and uses them to perform (implicitly) products with Q and QT.

No off-the-shelf solvers allowed. In particular you must implement yourself the thin QR factorization, and the computational cost of your implementation should scale linearly with the largest dimension of the matrix X.
