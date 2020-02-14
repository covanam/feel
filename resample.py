import pickle
import numpy as np
import os
import cv2

sampled_data = np.empty((1824 * 100, 3), dtype=np.float32)

index = 0
for filename in os.listdir('data/countryside'):
    filename = os.path.join('data/countryside', filename)
    x = cv2.imread(filename)
    x = x.reshape((-1, 3))
    choices = np.random.choice(x.shape[0], 100)
    for c in choices:
        sampled_data[index] = x[c, :]
        index += 1
        print(index)

for filename in os.listdir('data/metropolitian'):
    filename = os.path.join('data/metropolitian', filename)
    x = cv2.imread(filename)
    x = x.reshape((-1, 3))
    choices = np.random.choice(x.shape[0], 100)
    for c in choices:
        sampled_data[index] = x[c, :]
        index += 1
        print(index)

if index != 182400:
    raise RuntimeWarning('???')

with open('data/sampled.pkl', 'wb') as f:
    pickle.dump(sampled_data, f)
