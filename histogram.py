import cv2
import numpy as np
import os
import pickle
import time


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


def preprocess_data():
    data = np.empty((1824, 10, 4), dtype=np.dtype)
    label = np.empty(1824, dtype=np.int8)
    label[:891] = 1  # countryside #1
    label[891:] = 0
    index = 0
    for filename in os.listdir('data/countryside'):
        filename = os.path.join('data/countryside', filename)
        x = cv2.resize(cv2.imread(filename), dsize=(224, 224)).astype(np.float32)
        data[index] = kmean(x)
        print(index)
        index += 1

    for filename in os.listdir('data/metropolitian'):
        filename = os.path.join('data/metropolitian', filename)
        x = cv2.resize(cv2.imread(filename), dsize=(224, 224)).astype(np.float32)
        data[index] = kmean(x)
        index += 1

    if index != 1824:
        raise RuntimeError('what')

    with open('data/data.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open('data/label.pkl', 'wb') as f:
        pickle.dump(label, f)


if __name__ == '__main__':
    preprocess_data()
