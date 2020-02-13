import pickle
import numpy as np

with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f).astype(np.float32)

sampled_data = np.empty((data.shape[0] * 64, 3), dtype=data.dtype)

index = 0
for i, sample in enumerate(data):
    choices = np.random.choice(64, 64, p=sample[:, 0])
    for c in choices:
        sampled_data[index] = sample[c, 1:]
        index += 1

if index != sampled_data.shape[0]:
    raise RuntimeError('????????')

with open('data/sampled.pkl', 'wb') as f:
    pickle.dump(sampled_data, f)
