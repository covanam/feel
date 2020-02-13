from model import Feel
import pickle
import numpy as np

model = Feel()
model.d = pickle.load(open('model/d', 'rb'))
with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f).astype(np.float32)
with open('data/label.pkl', 'rb') as f:
    label = pickle.load(f).astype(np.bool)

model.train(data, label, 100, lr=1e-3)
pickle.dump(model, open('model/model', 'wb'))
