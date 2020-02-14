import os
import cv2
from histogram import kmean
import numpy as np
import pickle


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
        print(index)

    if index != 1824:
        raise RuntimeError('what')

    with open('data/data.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open('data/label.pkl', 'wb') as f:
        pickle.dump(label, f)


if __name__ == '__main__':
    preprocess_data()
