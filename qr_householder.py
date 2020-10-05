import numpy as np
from utils import norm, row_to_column, column_to_row
import scipy.linalg.lapack as sy


def householder_vector(x):
    '''
    This function creates a householder_reflector that can be used to form H as I - 2*v*v' to zero first entry
    :param x: a column vector
    :return: a column vector useful to form H
    '''
    s = -np.sign(x[0]) * norm(x)
    v = np.copy(x)
    v[0] = v[0] - s
    v = np.true_divide(v, norm(v))
    return v, s


def qr_factorization1(x_):
    '''
    First version of the algorithm, the slowest and simplest one
    It returns the QR factorization of the matrix x_, inside the function it uses a copy of the matrix
    :param x_: the matrix we want to factorize
    :return: orthogonal matrix Q and upper triangular R
    '''
    x_ = np.copy(x_)
    [m, n] = x_.shape
    Q_ = np.identity(m)
    for j in range(n):
        col = row_to_column(x_[j:, j])
        v_, _ = householder_vector(col) # v_ is a column vector (3, 1)
        v_size = v_.shape[0]
        v_T = column_to_row(v_) # v_T is a row vector (1, 3)
        H = np.identity(v_size) - (2 * np.matmul(v_, v_T))  # colonna x riga
        x_[j:, j:] = np.matmul(H, x_[j:, j:])
        Q_[:, j:] = np.matmul(Q_[:, j:], H)
    R_ = x_
    return Q_, R_


def qr_factorization2(x_):
    '''
    Variant of the first algorithm, which uses the fast product Householder-vector.
    It returns the QR factorization of the matrix x_, inside the function it uses a copy of the matrix
    :param x_: the matrix we want to factorize
    :return: orthogonal matrix Q and upper triangular R
    '''
    x_ = np.copy(x_)
    [m, n] = x_.shape
    Q_ = np.identity(m)
    for j in range(min(m-1, n)):
        col = row_to_column(x_[j:, j])
        v_, s_ = householder_vector(col) # v_ is a column vector (3, 1)
        v_T = column_to_row(v_) # v_T is a row vector (1, 3)
        x_[j, j] = s_
        x_[j+1:, j] = 0
        x_[j:, j+1:] = x_[j:, j+1:] - 2 * np.matmul(v_, np.matmul(v_T, x_[j:, j+1:]))
        Q_[:, j:] = Q_[:, j:] - np.matmul(Q_[:, j:], 2*np.matmul(v_, v_T))
    R_ = x_
    return Q_, R_


def qr_factorization3(x_):
    '''
    Last variant of the algorithm, which does not form the matrix Q, but stores the v's
    It returns the Householder vectors instead of the matrix Q, inside the function it uses a copy of the matrix
    :param x_: the matrix we want to factorize
    :return: Householder vectors V and upper triangular matrix R
    '''
    x_ = np.copy(x_)
    [m, n] = x_.shape
    V = []
    for j in range(min(m-1, n)):
        v_, s_ = householder_vector(row_to_column(x_[j:, j]))
        x_[j, j] = s_
        x_[j+1:, j] = 0
        x_[j:, j+1:] = x_[j:, j+1:] - 2 * np.matmul(v_, np.matmul(column_to_row(v_), x_[j:, j+1:]))
        V.append(v_)

    R_ = x_[:n, :]
    return V, R_


def qr_method(A_, b_):
    V, R = qr_factorization3(A_)
    m, n = A_.shape
    x = np.copy(b_)
    for j, vi in enumerate(V):
        aux1 = np.matmul(column_to_row(vi), x[j:])  # vi^T * b
        vi.shape = (vi.shape[0], 1)
        aux2 = 2 * np.matmul(vi, aux1)  # 2*vi*vi^T
        x[j:] = x[j:] - aux2  # b - 2*vi*vi^T*b

    # R_inv = np.linalg.inv(R)
    R_inv = sy.dtrtri(R)[0]
    x = np.matmul(R_inv, x[:n])
    return x
