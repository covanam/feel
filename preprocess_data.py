import os
import cv2
from histogram import kmean
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--image_size', default='224')
parser.add_argument('-k', default='5')
args = parser.parse_args()

im_size = int(args.image_size)
k = int(args.k)

data = np.empty((1824, k, 4), dtype=np.dtype)
label = np.empty(1824, dtype=np.int8)
label[:891] = 1  # countryside #1
label[891:] = 0
index = 0
for filename in os.listdir('data/countryside'):
    filename = os.path.join('data/countryside', filename)
    x = cv2.resize(cv2.imread(filename), dsize=(im_size, im_size)).astype(np.float32)
    data[index] = kmean(x, k=k)
    print(index)
    index += 1

for filename in os.listdir('data/metropolitian'):
    filename = os.path.join('data/metropolitian', filename)
    x = cv2.resize(cv2.imread(filename), dsize=(im_size, im_size)).astype(np.float32)
    data[index] = kmean(x, k=k)
    index += 1
    print(index)

if index != 1824:
    raise RuntimeError('what')

with open('data/data{:d}.pk'.format(k), 'wb') as f:
    pickle.dump(data, f)

with open('data/label.pkl', 'wb') as f:
    pickle.dump(label, f)
