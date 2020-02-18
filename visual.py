from model import Feel
import pickle
import cv2
import numpy as np

model = pickle.load(open('model/model', 'rb'))
alpha = model.alpha
d = model.d

d = [c for a, c in sorted(zip(alpha, d), key=lambda x: x[0], reverse=True)]
print(len(d))
im = np.empty((200, 1000, 3), np.uint8)
for i, color in enumerate(d):
    r = i // 10
    c = i % 10
    print(r, c)
    im[100*r:100*(r+1), 100*c:100*(1+c)] = color
cv2.imwrite('colors.png', im)
