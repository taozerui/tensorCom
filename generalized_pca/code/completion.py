import numpy
import pickle
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from math import pow, exp

class Latent(object):
    def __init__(self, feature):
        self.feature = feature
        self.Xhat = 0
        self.iter = 0

    def __len__(self):
        return self.feature

class NMF(Latent):
    def __init__(self, feature):
        super(NMF, self).__init__(feature)
        self.U = 0
        self.V = 0

    def __len__(self):
        return super(NMF, self).__len__()

    def fit(self, x, alpha, beta,
            learning_rate=0.1, max_iter=100, tol=1e-3, print_loss=False):
        if type(x) == numpy.ndarray:
            x = csc_matrix(x)

        feature = self.feature
        m, n = x.shape
        # mask matrix
        W = x.copy()
        W.data = np.ones(len(W.data))
        y = W.multiply(x)
        W = W.toarray()
        y = y.toarray()
        # init
        U = np.random.rand(m, feature)
        V = np.random.rand(n, feature)
        # main loop
        ## gradient descent
        loss = 0
        result = []
        for epoch in range(int(max_iter)):
            lossOld = loss
            ## update U
            gradient = - 2 * np.dot(y, V) + 2 * np.dot(W * np.dot(U, V.T), V) + alpha * U
            U -= learning_rate * gradient
            ## update V
            gradient = - 2 * np.dot(y.T, U) + 2 * np.dot(W.T * np.dot(V, U.T), U) + beta * V
            V -= learning_rate * gradient

            loss = np.sum((W * (np.dot(U, V.T)) - y) ** 2) + 0.5 * alpha * np.sum(U ** 2) + 0.5 * beta * np.sum(V ** 2)
            result.append(loss)
            if print_loss:
                print(f'Step {epoch + 1}, the loss is {loss:.3f}.')

            if abs(loss - lossOld) < tol:
                break

            if epoch + 1 == max_iter:
                print(f'Reach max iteration {int(max_iter)}!')

        self.U = U
        self.V = V
        self.Xhat = np.dot(U, V.T)
        self.iter = result
        return self

class LogisticPCA(Latent):
    def __init__(self, feature):
        super(LogisticPCA, self).__init__(feature)
        self.score = 0
        self.V = 0
        self.theta = 0
        self.prob = 0
        self.label = 0

    def __len__(self):
        return super(LogisticPCA, self).__len__()

    @staticmethod
    def _bregman(x, A, V, W):
        n, d = x.shape
        theta = np.dot(A, V)
        loss = 0
        for i in range(n):
            for j in range(d):
                if W[i, j] > 0:
                    #print(f'point ({i}, {j}), x: ({x[i, j]}), theta: ({theta[i, j]})')
                    foo = -x[i, j] * theta[i, j]
                    if foo > 10:
                        loss += foo
                    elif foo < -10:
                        loss += 0
                    else:
                        loss += np.log(1 + exp(foo))

        return loss

    def _fitScore(self, xi, ai, V, i, W):
        d = len(xi)
        max_iter = 100
        tol = 1e-3
        learning = 0.1
        for epoch in range(100):
            aOld = ai
            gradient = np.zeros(len(ai))
            for j in range(d):
                thetaj = np.dot(ai.reshape(1, -1), V[:, j].reshape(-1, 1))
                for k in range(self.feature):
                    gradient[k] += - W[i, j] * (xi[j] * V[k, j]) / (1 + exp(xi[j] * thetaj))
            ai -= learning * gradient
            if np.sum((ai - aOld) ** 2) < tol:
                break

        return ai

    def _fitV(self, xj, A, vj, j, W):
        n = len(xj)
        max_iter = 100
        tol = 1e-3
        learning = 0.1
        for epoch in range(100):
            vOld = vj
            gradient = np.zeros(len(vj))
            for i in range(n):
                thetai = np.dot(A[i, :].reshape(1, -1), vj.reshape(-1, 1))
                for k in range(self.feature):
                    gradient[k] += - W[i, j] * (xj[i] * A[i, k]) / (1 + exp(xj[i] * thetai))
            vj -= learning * gradient
            if np.sum((vj - vOld) ** 2) < tol:
                break

        return vj

    def fit(self, x, max_iter=100,
            tol=1e-4, print_loss=True):
        if type(x) == numpy.ndarray:
            x = csc_matrix(x)

        n, d = x.shape
        #x = 2 * x - 1
        feature = self.feature

        # mask matrix
        W = x.copy()
        W.data = np.ones(len(W.data))
        y = W.multiply(x)
        W = W.toarray()
        y = y.toarray()
        y = 2 * y - 1
        # init
        A = np.random.randn(n, feature)
        V = np.random.randn(feature, d)
        theta = np.dot(A, V)
        # main loop
        for epoch in range(int(max_iter)):
            thetaOld = theta
            ## update score
            for i in range(n):
                yi = y[i, :]
                ai = A[i, :]
                ai = self._fitScore(yi, ai, V, i, W)
                A[i, :] = ai.reshape(-1)
            ## update V
            for j in range(d):
                yj = y[:, j]
                vj = V[:, j]
                vj = self._fitV(yj, A, vj, j, W)
                V[:, j] = vj.reshape(-1)

            if print_loss:
                loss = self._bregman(y, A, V, W)
                print(f'Step {epoch+1}, the loss is {loss:.3f}.')

            theta = np.dot(A, V)
            if np.sum((thetaOld - theta) ** 2) < tol:
                break

            if epoch == max_iter - 1:
                print(f'Reach max iteration {int(max_iter)}!')

        self.score = A
        self.V = V
        self.theta = theta
        self.prob = self._prob()
        self.label = self._label()
        return self

    def _prob(self):
        theta = self.theta
        logit = lambda x: 1 / (1 + np.exp(-x))

        return logit(theta)

    def _label(self):
        theta = self.theta
        logit = lambda x: 1 / (1 + np.exp(-x))
        proba = logit(theta)
        label = proba.copy()
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        return label

if __name__ == '__main__':
    R=[
      [5,3,0,1],
      [4,0,0,1],
      [1,1,0,5],
      [1,0,0,4],
      [0,1,5,4]]

    R = np.array(R)
    K = 2

    model = NMF(K)
    model.fit(R, 0.02, 0.02, learning_rate=0.0001,
              max_iter=5e3, tol=1e-4, print_loss=True)
    print("原始的评分矩阵R为：\n",R)
    print("经过MF算法填充0处评分值后的评分矩阵R_MF为：\n",model.Xhat)
    plt.plot(model.iter)
    plt.show()
