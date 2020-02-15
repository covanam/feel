from model import Feel
import pickle
import cv2
import numpy as np

model = pickle.load(open('model/model', 'rb'))
alpha = model.alpha
d = model.d

significant_d = [c for a, c in sorted(zip(alpha, d), key=lambda x: x[0])]

countryside_colors = significant_d[-9:]
metropolitian_colors = significant_d[:9]

insignificant_d = [c for a, c in sorted(zip(alpha, d), key=lambda x: abs(x[0]))]
neutral_colors = insignificant_d[:9]

im = np.empty((300, 300, 3), np.uint8)
for i, color in enumerate(countryside_colors):
    r = i % 3
    c = i // 3
    im[100*r:100*(r+1), 100*c:100*(c+1)] = color
cv2.imwrite('countryside_colors.png', im)

for i, color in enumerate(metropolitian_colors):
    r = i % 3
    c = i // 3
    im[100*r:100*(r+1), 100*c:100*(c+1)] = color
cv2.imwrite('metropolitian_colors.png', im)

for i, color in enumerate(neutral_colors):
    r = i % 3
    c = i // 3
    im[100*r:100*(r+1), 100*c:100*(c+1)] = color
cv2.imwrite('neutral_colors.png', im)
