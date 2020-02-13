import pickle
import numpy as np
import cv2

with open('data/sampled.pkl', 'rb') as f:
    sampled = pickle.load(f)

criteria = (cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
_, _, d = cv2.kmeans(sampled, 256, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

with open('model/d', 'wb') as f:
    pickle.dump(d, f)

print(d[:10])
