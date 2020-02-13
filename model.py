import numpy as np
# from histogram import kmean


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Feel:
    def __init__(self, num_param=250, s=25.**2):
        self.s = s
        self.n_param = num_param
        self.alpha = np.random.normal(loc=0, scale=1 / num_param / s, size=num_param)
        self.d = np.ones((num_param, 3))
        self.b = np.zeros(1, dtype=np.float32)

    def __call__(self, x):
        pass

    def train(self, data, label, num_iter, lr=1e-3):
        for i in range(num_iter):
            loss, cache = self._forward(data, label)
            self._backward((loss, cache), label, lr=lr)
            # print(loss)

            y = cache[2]
            output = y > 0.5
            acc = np.sum(output == label) / y.shape[0]

            print(loss, acc)

    def _forward(self, x, label):
        temp = np.sum((self.d[np.newaxis, np.newaxis, :, :] - x[:, :, np.newaxis, 1:])**2, axis=3)
        temp /= self.s
        temp = np.exp(-0.5 * temp)

        gj = np.sum(self.alpha[np.newaxis, np.newaxis, :] * x[:, :, 0:1] * temp, axis=(1, 2))
        y = _sigmoid(gj + self.b)
        loss = np.sum(np.log(y[label])) + np.sum(np.log(1 - y[label == 0]))
        loss *= -1 / x.shape[0]

        cache = x, temp, y

        return loss, cache

    def _backward(self, output, t, lr=1e-3):
        loss, cache = output
        x, exp, y = cache

        grad = t * (y - 1) + (1 - t) * y

        dalpha = np.sum(x[:, :, 0:1] * exp, axis=1)
        dalpha *= grad[:, np.newaxis]
        dalpha = np.mean(dalpha, axis=0)

        temp = -(self.d[np.newaxis, np.newaxis, :, :] - x[:, :, np.newaxis, 1:]) / self.s
        dd = - np.sum(x[:, :, 0:1, np.newaxis] * exp[:, :, :, np.newaxis] * temp, axis=1)
        dd *= grad[:, np.newaxis, np.newaxis]
        dd = np.mean(dd, axis=0)
        dd *= self.alpha[:, np.newaxis]

        db = np.mean(grad)

        self.alpha -= dalpha * lr
        self.d -= dd * lr
        self.b -= db * lr
