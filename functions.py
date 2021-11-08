import numpy as np


def dist(x, y):
    if len(x.shape) == 1:
        x = np.array([x])

    if len(y.shape) == 1:
        y = np.array([y])

    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def initialize_partition_matrix(shape):
    base_array = np.random.random(shape)
    return (1 / np.sum(base_array, axis=1).reshape(-1, 1)) * base_array


def partition_matrix(X, centers, m):
    n_centers, dimension = centers.shape
    temp = np.transpose(
        np.array([dist(X, centers[i]) ** (2 / (m - 1)) for i in range(n_centers)])
    )
    mu = 1 / (np.sum(1 / temp, axis=1).reshape(-1, 1) * temp)
    return mu


def update_centers(X, mu, m):
    n_points, n_centers = mu.shape

    return np.array(
        [
            np.sum((mu[:, i] ** m).reshape(-1, 1) * X, axis=0) / np.sum(mu[:, i] ** m)
            for i in range(n_centers)
        ]
    )


def objective(X, centers, m, mu=None):
    n_centers, dimension = centers.shape

    if mu is None:
        mu = partition_matrix(X, centers, m=m)

    temp = np.transpose(np.array([dist(X, centers[i]) ** 2 for i in range(n_centers)]))

    return np.sum(mu ** m * temp)


def initialize_centers(X, n_centers):

    centers = np.array(
        [
            [
                np.random.uniform(low=X[:, i].min(), high=X[:, i].max())
                for i in range(X.shape[1])
            ]
            for j in range(n_centers)
        ]
    )

    return centers
