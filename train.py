from model import Feel
import pickle
import numpy as np

model = Feel()
model.d = pickle.load(open('model/d', 'rb'))
# model = pickle.load(open('model/model', 'rb'))
with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f).astype(np.float32)
with open('data/label.pkl', 'rb') as f:
    label = pickle.load(f).astype(np.bool)

train_data = np.concatenate((data[0:600], data[1224:]), axis=0)
train_label = np.concatenate((label[0:600], label[1224:]), axis=0)

val_data = data[600:1224]
val_label = label[600:1224]

del data
del label

model.train(train_data, train_label, 5000, 1e-3, val_data, val_label)
pickle.dump(model, open('model/model', 'wb'))
