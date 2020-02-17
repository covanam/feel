from model import Feel
import pickle
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_param', default='32')
parser.add_argument('-k', default='4')
parser.add_argument('-s', default='30')
parser.add_argument('--num_epoch', default='100')
parser.add_argument('--train_d', default='false')
args = parser.parse_args()

num_param = int(args.num_param)
k = int(args.k)
s = float(args.s)
num_epoch = int(args.num_epoch)
train_d = args.train_d == 'true'

d_filename = 'cache/d_n{:d}.pk'.format(num_param)
if not os.path.isfile(d_filename):
    # estimate d if haven't already
    with open('data/sampled.pkl', 'rb') as f:
        sampled = pickle.load(f)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, _, d = cv2.kmeans(sampled, num_param, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    pickle.dump(d, open(d_filename, 'wb'))
else:
    d = pickle.load(open(d_filename, 'rb'))

model = Feel(num_param, s**2)
model.d = d
# model = pickle.load(open('model/model', 'rb'))

# load data
with open('data/data{:d}.pk'.format(k), 'rb') as f:
    data = pickle.load(f).astype(np.float32)
with open('data/label.pkl', 'rb') as f:
    label = pickle.load(f).astype(np.bool)

train_data = np.concatenate((data[0:600], data[1224:]), axis=0)
train_label = np.concatenate((label[0:600], label[1224:]), axis=0)

val_data = data[600:1224]
val_label = label[600:1224]

del data
del label

model.train(train_data, train_label, num_epoch, 1e-3, train_d, val_data, val_label)
filename = 'model/model_h{0:d}_k{1:d}_s{2:.0f}'.format(num_param, k, s)
pickle.dump(model, open(filename, 'wb'))
