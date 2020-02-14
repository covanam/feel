import numpy as np
import histogram
from histogram import kmean
import cv2
import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Feel:
    def __init__(self, num_param=256, s=32.**2):
        self.s = s
        self.alpha = np.zeros(shape=num_param)
        self.d = np.random.random((num_param, 3)) * 255
        self.b = np.zeros(1, dtype=np.float32)

    def __call__(self, x):
        x = cv2.resize(x, (224, 224)).astype(np.float32)
        x = kmean(x, k=32)

        p = x[:, 0:1]
        exp = np.sum((self.d[np.newaxis, :, :] - x[:, np.newaxis, 1:]) ** 2, axis=2)
        exp /= self.s
        exp = np.exp(-0.5 * exp)

        gj = np.sum(self.alpha[np.newaxis, :] * p * exp)
        y = _sigmoid(gj + self.b)

        return y

    def train(self, data, label, num_iter, lr=1e-3, val_data=None, val_label=None):
        p = data[:, :, 0:1]
        exp = np.sum((self.d[np.newaxis, np.newaxis, :, :] - data[:, :, np.newaxis, 1:]) ** 2, axis=3)
        exp /= self.s
        exp = np.exp(-0.5 * exp)

        val_exp = np.sum((self.d[np.newaxis, np.newaxis, :, :] - val_data[:, :, np.newaxis, 1:]) ** 2, axis=3)
        val_exp /= self.s
        val_exp = np.exp(-0.5 * val_exp)
        val_p = val_data[:, :, 0:1]

        for i in range(num_iter):
            loss, y = self._forward(p, exp, label)
            self._backward((loss, y), label, p, exp, lr=lr)
            # print(loss)

            output = y > 0.5
            acc = np.sum(output == label) / y.shape[0]

            # print('training:', loss, acc)
            if True: #  i % 20 == 0:
                val_loss, y = self._forward(val_p, val_exp, val_label)
                output = y > 0.5
                val_acc = np.sum(output == val_label) / output.shape[0]
                print('{:.4f}, {:.4f}'.format(loss, val_loss))
                print('{:.2f}, {:.2f}'.format(acc, val_acc))

    def _forward(self, p, exp, label):
        gj = np.sum(self.alpha[np.newaxis, np.newaxis, :] * p * exp, axis=(1, 2))
        y = _sigmoid(gj + self.b)
        loss = np.sum(np.log(y[label])) + np.sum(np.log(1 - y[label == 0]))
        loss *= -1 / p.shape[0]

        return loss, y

    def _backward(self, output, t, p, exp, lr=1e-3):
        loss, y = output

        grad = t * (y - 1) + (1 - t) * y

        dalpha = np.sum(p * exp, axis=1)
        dalpha *= grad[:, np.newaxis]
        dalpha = np.mean(dalpha, axis=0)

        db = np.sum(grad)

        self.alpha -= dalpha * lr
        self.b -= db * lr
