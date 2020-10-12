import numpy as np
from utils import transpose_matrix, norm

def conjugate_gradient(A, b, x0 = None, eps = None, maxIter = 1000):
    '''
    Performs conjugate gradient method to the function f(x) = 1/2(A*Ax) - x*A*b
    :param A: a matrix mxn
    :param b: a column vector nx1
    :param x0: starting point, if None the 0 vector is used as default starting point
    :param eps: (optional, default value 1e-5) the accuracy in the stopping criterion
    :param maxIter:  (optional, default value 1000): the maximum number of iterations
    :return: [x, status, ite]: 
    :  - x (mx1 real column vector): it solves ||gradient(f(x))|| = A*Ax - A*b = 0
    :  - status (string): a string describing the status of the algorithm at
    :    termination 
    :    = 'optimal': the algorithm terminated having proven that x is an optimal solution, i.e., 
    :                 the norm of the gradient at x is less than the required threshold
    :    = 'finished': the algorithm terminated in m iterations since no treshold of accuracy is required
    :    = 'stopped': the algorithm terminated having exhausted the maximum number of iterations
    :  - ite: number of iterations executed by the algorithm
    '''
    if x0 is None:
        x = np.zeros(A.shape[1])
    else:
        x = x0
        
    r =  b  - np.matmul(A, x)
    g_0 = np.matmul(transpose_matrix(A), r)
    d = g_0
    g = g_1 = g_0
    i = 1
    while True:
        if i>1:
            g_2 = g_1
            g_1 = g
            beta = -np.divide(np.square(norm(g_1)),np.square(norm(g_2)))
            d = g_1 - beta*d   
            #print("Space used ", r.nbytes+d.nbytes+x.nbytes+g.nbytes+g_1.nbytes+g_2.nbytes)
        Ad = np.matmul(A, d) 
        alpha = np.divide(np.square(norm(g_1)),np.square(norm(Ad)))
        x = x + alpha*d
        r = r - alpha*Ad
        g = np.matmul(transpose_matrix(A), r)
        ng = norm(g)
        
        if eps is None: 
            # no stopping condition, we end up in m iterations or when the norm of the gradient is zero
            if not np.any(g):
                status = "optimal"
                break
            if i > A.shape[1]:
                status = "finished"
                break
        else:
            # check accuracy for stopping condition
            if ng <= eps:
                status = "optimal"
                break
            if i > maxIter:
                status = "stopped"
                break
            
        i = i+1
        

    return x, status, i-1               



