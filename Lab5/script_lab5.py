"""
Introduction to Machine Learning

Lab 5: Compressive Sensing

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab4.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): Zihao Liu
Student ID(s): 2024201598
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import scipy.linalg


# don't add any other packages


# Task 1: Implement Sparse Data Generation Function
def sparse_data(dictionary: np.ndarray, sparsity: int = 2, n: int = 1, random_seed: int = 42) -> np.ndarray:
    """
    Implement PCA via eigen-decomposition
    :param dictionary: a dictionary matrix with (D, K), D is the dimension of data, K is the number of atoms/columns in
    the dictionary
    :param sparsity: the number of nonzero coefficients used to construct the data
    :param n: the number of samples in the data
    :param random_seed: the random seed used to generate coefficients.
    :return:
        the zero-mean data with size (N, D)
    """
    # TODO: Replace the code below, Implement the data generation pipeline
    rng = np.random.RandomState(random_seed)
    D, K = dictionary.shape

    coeff = np.zeros((n,K))
    for i in range(n):
        idx = rng.choice(K,size=sparsity,replace=False)
        coeff[i,idx] = rng.randn(sparsity)

        xs = coeff @ dictionary.T
        xs = xs - np.mean(xs,axis=0,keepdims=True)
        
    return xs


# Task 2: Implement the random projection
def random_projection(xs: np.ndarray, dim: int = 10, sense_type: str = 'normal', random_seed: int = 10) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Implement data whitening via the method in Lecture 2 or PCA
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param dim: the dimension of output
    :param sense_type: 'normal' or 'bernoulli', determining the type of random projection matrix
    :param random_seed: the random seed used to generate the random projection matrix
    :return:
        ys: the data yield normal distribution, with size (N, D)
        proj: the random projection matrix
    """
    # TODO: Replace the code below, Implement the random projection generation step
    rng = np.random.RandomState(random_seed)
    D = xs.shape[1]
    
    if sense_type == 'normal':
        proj = rng.randn(D, dim) / np.sqrt(dim)
    elif sense_type == 'bernoulli':
        proj = rng.choice([-1.0, 1.0], size=(D, dim))
    else:
        raise ValueError("sense_type must be 'normal' or 'bernoulli'")
    
    ys = xs @ proj
    return ys, proj
        


# Task 3: Implement the data recovery algorithm
def data_recovery(ys: np.ndarray, dictionary: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """
    Implement the data recovery algorithm (Hint: Recall the Lasso algorithm you learned before)
    :param ys: the random projection result with size (N, dim)
    :param dictionary: a dictionary matrix with (D, K), D is the dimension of data, K is the number of atoms/columns in
    the dictionary
    :param proj: the random projection matrix with size (D, dim)
    :return:
        xs: the recovery data matrix with size (N, D)
    """
    # TODO: implement the data recovery algorithm
    # Effective sensing matrix: y^T = P^T D a
    A = proj.T @ dictionary   # shape: (dim, K)
    
    # L = largest eigenvalue of A^T A = spectral norm(A)^2, Find tao
    svals = np.linalg.svd(A, compute_uv=False)
    L = (svals[0] ** 2) if len(svals) > 0 and svals[0] > 1e-12 else 1.0

    lam = 1e-2
    max_iter = 200

    def soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

    N = ys.shape[0]
    K = dictionary.shape[1]
    coeffs = np.zeros((N, K))
    
    for i in range(N):
        b = ys[i]          # shape: (dim,)
        a = np.zeros(K)

        for _ in range(max_iter):
            grad = A.T @ (A @ a - b)
            a = soft_threshold(a - grad / L, lam / L)

        coeffs[i] = a

    xs = coeffs @ dictionary.T   # (N, K) @ (K, D) = (N, D)
    return xs


# Task 4: Visualize the covariance matrix
def visualization_cov(xs: np.ndarray):
    """
    Visualize the covariance matrix of data
    :param xs: a data matrix with size (N, D)
    :return: (visualize)
        cov: the covariance matrix with size (D, D)
    """
    # TODO: implement the computation and visualization of covariance matrix
    # ensure zero-mean along feature dimension
    xs_centered = xs - np.mean(xs, axis=0, keepdims=True)

    # covariance matrix: shape (D, D)
    if xs.shape[0] > 1:
        cov = xs_centered.T @ xs_centered / (xs.shape[0] - 1)
    else:
        cov = np.zeros((xs.shape[1], xs.shape[1]))

    plt.imshow(cov, cmap='viridis', aspect='auto')
    plt.colorbar()
    return cov


# Testing script
if __name__ == '__main__':
    dictionary = scipy.linalg.hadamard(128, dtype=float)
    print(dictionary)
    data = sparse_data(dictionary)
    plt.figure()
    visualization_cov(data)
    plt.title('real data cov')
    plt.savefig('data_cov.png')
    plt.close('all')

    for sense_type in ['normal', 'bernoulli']:
        for dim in [4, 8, 16, 32]:
            ys, proj = random_projection(xs=data, dim=dim, sense_type=sense_type)
            xs = data_recovery(ys=ys, dictionary=dictionary, proj=proj)
            print('SenseType={}, Dim={}, MSE={}'.format(sense_type, dim, np.sum((data-xs)**2)))
            plt.figure()
            visualization_cov(xs)
            plt.title('est data cov')
            plt.savefig('est_data_cov_{}_{}.png'.format(sense_type, dim))
            plt.close('all')

            plt.figure()
            visualization_cov(ys)
            plt.title('cs cov')
            plt.savefig('cs_data_cov_{}_{}.png'.format(sense_type, dim))
            plt.close('all')
