"""
Introduction to Machine Learning

Lab 5: Matrix factorization and linear dimensionality reduction

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
# don't add any other packages


# data simulator and testing function (Don't change them)
def zero_mean_point_cloud_simulator(n_pts: int = 50,
                                    r_seed: int = 42) -> dict:
    """
    Simulate a set of zero-mean 2D points with Gaussian noise or outliers
    :param n_pts: the number of 2D points
    :param r_seed: the random seed
    :return:
        a dictionary containing the points with Gaussian noise and those with outliers, respectively
    """
    x = 4 * (np.random.RandomState(r_seed).rand(n_pts, 1) - 0.5)
    y = 0.4 * x
    data = np.concatenate((x, y), axis=1)
    pts1 = data + 0.1 * np.random.RandomState(r_seed).randn(n_pts, 2)
    pts2 = data + 0.01 * np.random.RandomState(r_seed).randn(n_pts, 2)
    idx = np.random.RandomState(r_seed).permutation(n_pts)
    n_noise = int(0.2 * n_pts)
    pts2[idx[:n_noise], :] = np.random.RandomState(r_seed).randn(n_noise, 2) + np.array([0.5, 1.5]).reshape((1, 2))
    return {'gauss': pts1, 'outlier': pts2}


def visualization_pts(pts: np.ndarray, label: str, point_type: str):
    plt.plot(pts[:, 0], pts[:, 1], point_type, label=label)


def visualization_line(v: np.ndarray, label: str, line_type: str):
    xs = 5 * (np.arange(0, 100) / 100 - 0.5)
    ys = v[1] / v[0] * xs
    plt.plot(xs, ys, line_type, label=label)


# Task 1: Implement PCA via eigen-decomposition
def pca(xs: np.ndarray, n_pc: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement PCA via eigen-decomposition
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space
    :param n_pc: the number of principal components we would like to output
    :return:
        the matrix containing top-k principal components, with size (D, n_pc)
        the vector indicating the top-k eigenvalues, with size (n_pc)
        the data recovered from the projections along the principal components, with size (N, D)
        the zero-mean data with size (N, D)
    """
    # TODO: Change the code below and implement your PCA algorithm
    x_mean = np.mean(xs, axis=0, keepdims=True)
    x_centered = xs - x_mean

    cov = (x_centered.T @ x_centered) / xs.shape[0]

    eigvals, eigvecs = np.linalg.eigh(cov) # 特征值，特征向量

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pcs = eigvecs[:, :n_pc]
    top_eigvals = eigvals[:n_pc]

    scores = x_centered @ pcs
    x_recovered = scores @ pcs.T

    return pcs,top_eigvals,x_recovered,x_centered


# Task 2: Implement data whitening via the method in Lecture 2 and the PCA-based method in Lecture 5
def data_whitening(xs: np.ndarray) -> np.ndarray:
    """
    Implement data whitening via the method in Lecture 2 or PCA
    :param xs: the data matrix with size (N, D), N is the number of samples
    :return:
        ys: the data yield normal distribution, with size (N, D)
    """
    # TODO: Change the code below and implement your data whitening method (Hint: you can call the above PCA function)
    pcs, lambdas, _, x_centered = pca(xs, n_pc=xs.shape[1])

    eps = 1e-12
    lambdas = np.maximum(lambdas, eps)

    ys = x_centered @ pcs @ np.diag(1.0 / np.sqrt(lambdas)) @ pcs.T
    return ys

# Task 3: Try to develop your own method to achieve robust PCA (the method may not be the state-of-the-art, but doable)
def hard_thresholding(x: np.ndarray, ratio: float) -> np.ndarray:
    """
    The hard-thresholding operator
    :param x: input array with arbitrary size
    :param ratio: the ratio of nonzero elements
    :return:
        y = x,  if |x| > a threshold
            0,  otherwise
    """
    # TODO: change the code below,
    #  implement a hard-thresholding method (given a ratio of nonzero elements, determine the threshold adaptively)
    num_keep = int(ratio * x.size)
    if num_keep <= 0:
        return np.zeros_like(x)
    if num_keep >= x.size:
        return x.copy()

    abs_x = np.abs(x).reshape(-1)
    threshold = np.partition(abs_x, -num_keep)[-num_keep]

    y = x.copy()
    y[np.abs(y) < threshold] = 0
    return y


def robust_pca_hard(xs: np.ndarray, n_pc: int = 2, n_alt: int = 100,
                    ratio_nz: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement your own algorithm to solve the robust PCA problem via
    optimizing the low-rank factorization of data matrix (X in R^{N x D}) explicitly, i.e.,

    min_{L, S} ||X - (L + S)||_F^2
    s.t. rank(L) <= n_pc, ||S||_0 < ratio_nz * (N * D)

    Hint: you may want to solve L and S in an alternating optimization manner:
    1) Fix L and solve
        L = argmin_L ||X - (L + S)||_F^2
        s.t. rank(L) <= n_pc
    2) Fix S and solve
        S = argmin_S ||X - (L + S)||_F^2,
        s.t.. ||S||_0 < ratio_nz * (N * D)

    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space.
    :param n_pc: the number of principal components we would like to output.
    :param n_alt: the number of steps for alternating optimization.
    :param ratio_nz: the ratio of non-zero elements in the whole matrix.
    :return:
        the matrix containing top-k principal components, with size (D, n_pc)
        the vector indicating the top-k eigenvalues, with size (n_pc)
        the data recovered from the projections along the principal components, with size (N, D)
        the zero-mean data with size (N, D)
    """
    # TODO: Change the code below, implement the Robust PCA according to the comments
    x_mean = np.mean(xs, axis=0, keepdims=True)
    x_centered = xs - x_mean

    # initialize sparse component
    s = np.zeros_like(x_centered)
    l = np.zeros_like(x_centered)

    # alternating optimization
    for _ in range(n_alt):
        # P1: update low-rank part by PCA on (X - S)
        y = x_centered - s
        pcs, lambdas, l, _ = pca(y, n_pc=n_pc)

        # P2: update sparse part by hard-thresholding residual
        residual = x_centered - l
        s = hard_thresholding(residual, ratio_nz)

    # extract principal components from the final low-rank part
    pcs, lambdas, xhat, xzm = pca(l, n_pc=n_pc)

    return pcs, lambdas, xhat, xzm


# Task 4: Suppose that you are a data attacker. Because of limited budgets, you can only add two outliers
# Try to design a "data poisoning" strategy to change the covariance of the data as much as possible.
def coupled_outlier_poisoning(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two outliers "x1" and "x2", with constraints ||x1||_2 = ||x2||_2 = 1 and x1 + x2 = 0
    :param xs: a data matrix with size (N, D), N is the number of samples
    :return:
        the outliers with size (2, D)
        the new data matrix with the outlier, with size (N+2, D)
    """
    # TODO: change the code below and implement the data poisoning method
    x_mean = np.mean(xs, axis=0, keepdims=True)
    x_centered = xs - x_mean

    cov = (x_centered.T @ x_centered) / xs.shape[0]

    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, np.argmin(eigvals)]

    u = u / (np.linalg.norm(u) + 1e-12)

    outliers = np.stack([u, -u], axis=0)

    xs_new = np.concatenate([x_centered, outliers], axis=0)

    return outliers, xs_new


# Task 5: implement the NMF algorithm
def nonnegative_matrix_factorization(xs: np.ndarray,
                                     rank: int,
                                     num_iter: int = 100,
                                     seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the nonnegative matrix factorization

    min_{U, V} ||X - UV^T||_F^2

    s.t. U in [0, inf]^{(N, r)} and V in [0, inf]^{(D, r)}

    :param xs: a data matrix with size (N, D), N is the number of samples
    :param rank: the rank of U and V
    :param num_iter: the number of iterations
    :param seed: the random seed of initialization
    :return:
        U in [0, inf]^{(N, r)}
        V in [0, inf]^{(D, r)}
        hat{X} = UV^T
    """
    us = np.random.RandomState(seed=seed).rand(xs.shape[0], rank)
    vs = np.random.RandomState(seed=seed + 2).rand(xs.shape[1], rank)
    # TODO: change the code below and implement the NMF algorithm
    eps = 1e-12
    xs_nonneg = np.maximum(xs, 0)

    for _ in range(num_iter):
        # update U
        us *= (xs_nonneg @ vs) / (us @ (vs.T @ vs) + eps)

        # update V
        vs *= (xs_nonneg.T @ us) / (vs @ (us.T @ us) + eps)

    xhat = us @ vs.T
    return us, vs, xhat


# Testing script
if __name__ == '__main__':
    data = zero_mean_point_cloud_simulator()
    for noise_type in data.keys():
        vs1, lambdas1, xhat1, xs1 = pca(data[noise_type], n_pc=1)
        vs2, lambdas2, xhat2, _ = robust_pca_hard(data[noise_type], n_pc=1, ratio_nz=0.1)
        xhat3 = data_whitening(data[noise_type])

        plt.figure()
        visualization_pts(xs1, label='data points', point_type='g.')
        visualization_pts(xhat1, label='pca', point_type='rx')
        visualization_pts(xhat2, label='rpca', point_type='bx')
        visualization_line(v=vs1, label='pca v1', line_type='r:')
        visualization_line(v=vs2, label='rpca v1', line_type='b:')
        visualization_line(v=np.array([1, 0.4]), label='real pc', line_type='g:')
        result = 'PCA vs RPCA: {} noise'.format(noise_type)
        plt.title(result)
        plt.legend()
        plt.savefig('result_{}.png'.format(noise_type))
        plt.close('all')

        plt.figure()
        visualization_pts(data[noise_type], label='before whitening', point_type='g.')
        visualization_pts(xhat3, label='after whitening', point_type='rx')
        plt.legend()
        plt.axis('equal')
        plt.savefig('whitening_{}.png'.format(noise_type))
        plt.close('all')

    vs1, lambdas1, xhat1, xs1 = pca(data['gauss'], n_pc=1)
    outliers, data_noisy = coupled_outlier_poisoning(data['gauss'])
    print(data['gauss'].shape, data_noisy.shape)
    vs2, lambdas2, xhat2, _ = pca(data_noisy, n_pc=1)
    plt.figure()
    visualization_pts(data['gauss'], label='data points', point_type='g.')
    visualization_pts(outliers, label='outlier', point_type='k*')
    visualization_pts(xhat1, label='PCA before poisoning', point_type='rx')
    visualization_pts(xhat2, label='PCA after poisoning', point_type='bx')
    visualization_line(v=vs1, label='v1 before poisoning', line_type='r:')
    visualization_line(v=vs2, label='v1 after poisoning', line_type='b:')
    visualization_line(v=np.array([1, 0.4]), label='real pc', line_type='g:')
    result = 'Covariance poisoning'
    plt.title(result)
    plt.legend()
    plt.axis('equal')
    plt.savefig('poisoning_pca.png')
    plt.close('all')

    data_mat = np.random.RandomState(seed=42).rand(100, 50)
    for r in [5, 10, 20, 30, 40]:
        u_mat, v_mat, data_approx = nonnegative_matrix_factorization(xs=data_mat, rank=r, num_iter=100, seed=1)
        error = np.sum(np.abs(data_mat - data_approx)) / np.sum(data_mat)
        print('Rank-{} NMF approximation RMAE={}'.format(r, error))
