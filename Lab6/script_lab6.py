"""
Introduction to Machine Learning

Lab 6: Nonlinear dimensionality reduction

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab6.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s):
Student ID(s):
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
# don't add any other packages


# data simulator and testing function (Don't change them)
def simulate_3d_manifold(n_pts: int = 500, noise_level: float = 0.01, r_seed: int = 42) -> dict:
    """
    Simulate a set of 3D points lying on a manifold, the manifold is a 2D geometry embedded in the 3D space.
    :param n_pts: the number of 3D points
    :param r_seed: the random seed
    :param noise_level: the standard deviation of Gaussian noise
    :return:
        a dictionary containing the 3D points with Gaussian noise and their 2D latent codes.
    """

    t1 = 5 * np.pi / 3 * np.random.RandomState(r_seed).rand(n_pts, 1)
    t2 = 5 * np.pi / 3 * np.random.RandomState(1).rand(n_pts, 1)
    latent_code = np.concatenate((t1, t2), axis=1)
    x1 = 3 + np.cos(t1) * np.cos(t2)
    x2 = 3 + np.cos(t1) * np.sin(t2)
    x3 = np.sin(t1)
    data = np.concatenate((x1, x2, x3), axis=1) + noise_level * np.random.RandomState(r_seed).randn(n_pts, 3)
    return {'3d': data, '2d': latent_code}


def visualization_3d_pts(pts3d: np.ndarray, prefix: str = 'data'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    plt.savefig('{}_3d.png'.format(prefix))
    plt.close()


def visualization_2d_pts(pts2d: np.ndarray, prefix: str = 'data'):
    plt.figure(figsize=(12, 12))
    plt.scatter(pts2d[:, 0], pts2d[:, 1])
    plt.savefig('{}_2d.png'.format(prefix))
    plt.close()


# Task 1: Implement Kernel PCA
def distance_matrix(xs: np.ndarray, distance_type: str = 'L2') -> np.ndarray:
    """
    Construct a N x N distance matrix from a data matrix with size (N, D)
    :param xs: a data matrix with size (N, D)
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        a distance matrix with size (N, N)
    """
    # TODO: Change the code below
    return np.zeros((xs.shape[0], xs.shape[0]))


def kernel(x: np.ndarray, k_type: str = 'rbf', bandwidth: float = 1) -> np.ndarray:
    """
    Implement typical kernel functions
    1) RBF kernel: k(x, y) = exp(-||x - y||_2^2 / bandwidth)
    2) Linear kernel: k(x, y) = <x, y>

    Hint: Recall your Lab work 4

    :param x: a set of samples with size (N, D), where N is the number of samples, D is the dimension of features
    :param k_type: the type of kernels, including 'rbf', 'linear'
    :param bandwidth: the hyperparameter controlling the width of rbf kernels
    :return:
        return a matrix with size (M, N)
    """
    # TODO: Change the code below
    return np.ones((x.shape[0], x.shape[0]))


def kernel_pca(xs: np.ndarray, d: int, k_type: str = 'rbf', bandwidth: float = 1) -> np.ndarray:
    """
    Implement kernel PCA
    :param xs: the data matrix with shape (N, D)
    :param d: the number of dimensions after dimensionality reduction
    :param k_type: the type of kernels, including 'rbf', 'linear'
    :param bandwidth: the hyperparameter controlling the width of rbf kernels
    :return:
    """
    # TODO: Change the code below
    return xs[:, -d:]


# Task 2: Construct a K-NN graph from data points
def construct_knn_graph(xs: np.ndarray, k: int = 5, distance_type: str = 'L2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a K-NN graph from the data points and output the adjacency matrix and the index matrix
    :param xs: a data matrix with (N, D), N is the number of samples, D is the dimension of sample space
    :param k: the number of principal components we would like to output
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        an adjacency matrix with size (N, N)
        an index matrix with size (N, k), the n-th row contains the indices of the neighbors of the n-th sample.
    """
    # TODO: change the code below
    return np.zeros((xs.shape[0], xs.shape[0])), np.zeros((xs.shape[0], k))


# Task 2: Implement the Locally Linear Embedding algorithm
def locally_linear_embedding(xs: np.ndarray, k: int = 5, dim: int = 2, distance_type: str = 'L2') -> np.ndarray:
    """
    Implement the locally linear embedding algorithm
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param k: the number of neighbors per sample in the K-NN graph
    :param dim: the dimension of latent code, where dim < D
    :param distance_type: the type of the distance, which can be "L2" or "L1",
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :return:
        ys: the latent codes of the data, with size (N, dim)
    """
    # TODO: change the code below
    return np.zeros((xs.shape[0], dim))


# Task 3: Implement the Laplacian eigenmap algorithm
def laplacian_eigenmaps(xs: np.ndarray, k: int = None, dim: int = 2,
                        normalize: bool = True, bandwidth: float = 4) -> np.ndarray:
    """
    Implement the Laplacian Eigenmap algorithm
    :param xs: the data matrix with size (N, D), N is the number of samples
    :param k: the number of neighbors per sample in the K-NN graph, if k is None, we obtain a fully-connected graph
    :param dim: the dimension of latent code, where dim < D
        L2 means d_ij = ||xi - xj||_2, while L1 means d_ij = ||xi - xj||_1
    :param normalize: use normalized Laplacian or not
    :param bandwidth: the bandwidth of kernel for computing the similarity matrix
    :return:
        ys: the latent codes of the data, with size (N, dim)
    """
    # TODO: change the code below
    return np.zeros((xs.shape[0], dim))


# Testing script
if __name__ == '__main__':
    data = simulate_3d_manifold()
    visualization_3d_pts(data['3d'], prefix='data')
    visualization_2d_pts(data['2d'], prefix='data')
    for h in [0.01, 0.1, 1, 10, 100]:
        z0 = kernel_pca(xs=data['3d'], d=2, k_type='rbf', bandwidth=h)
        visualization_2d_pts(z0, prefix='KPCA_rbf_{}'.format(int(np.log10(h))))
    z1 = kernel_pca(xs=data['3d'], d=2, k_type='linear')
    visualization_2d_pts(z1, prefix='KPCA_linear')

    for k in [3, 5, 10, 25, 50, 100, 200]:
        z1 = locally_linear_embedding(xs=data['3d'], k=k)
        visualization_2d_pts(z1, prefix='LLE_{}'.format(k))

    for k in [3, 5, 10, 25, 50, 100, 200, None]:
        for normalize in [True, False]:
            z2 = laplacian_eigenmaps(xs=data['3d'], k=k, normalize=normalize)
            if k is None:
                prefix = 'LE_full_{}'.format(normalize)
            else:
                prefix = 'LE_{}_{}'.format(k, normalize)
            visualization_2d_pts(z2, prefix=prefix)
