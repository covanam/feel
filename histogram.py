import cv2
import numpy as np


def kmean(x: np.ndarray, k=10):
    x = x.reshape((-1, 3))

    result = np.empty((k, 4), dtype=x.dtype)
    # first column: probabilities of a cluster
    # remaining 3 columns: mean of cluster

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, result[:, 1:] = cv2.kmeans(x, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)

    unique, count = np.unique(labels, return_counts=True)

    for i, n in zip(unique, count):
        result[i, 0] = n / x.shape[0]

    return result
