import ipdb
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from scipy.linalg import solve

class GeneralizedPCA(object):
    def __init__(self, feature):
        self.feature = feature
        self.score = 0
        self.V = 0
        self.theta = 0

    def __len__(self):
        return self.feature

    def _fitScore(self, xi, ai, V):
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
                    gradient[k] += - (xi[j] * V[k, j]) / (1 + exp(xi[j] * thetaj))
            ai -= learning * gradient
            if np.sum((ai - aOld) ** 2) < tol:
                break

        return ai

    def _fitV(self, xj, A, vj):
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
                    gradient[k] += - (xj[i] * A[i, k]) / (1 + exp(xj[i] * thetai))
            vj -= learning * gradient
            if np.sum((vj - vOld) ** 2) < tol:
                break

        return vj

    @staticmethod
    def _bregman(x, A, V):
        n, d = x.shape
        theta = np.dot(A, V)
        loss = 0
        for i in range(n):
            for j in range(d):
                loss += np.log(1 + exp(- x[i, j] * theta[i, j]))

        return loss

    def fit(self, x, max_iter=100,
            tol=1e-4, print_loss=True):
        n, d = x.shape
        x = 2 * x - 1
        feature = self.feature

        # init
        A = np.random.randn(n, feature)
        V = np.random.randn(feature, d)
        theta = np.dot(A, V)
        # main loop
        for epoch in range(int(max_iter)):
            thetaOld = theta
            ## update score
            for i in range(n):
                xi = x[i, :]
                ai = A[i, :]
                ai = self._fitScore(xi, ai, V)
                A[i, :] = ai.reshape(1, -1)
            ## update V
            for j in range(d):
                xj = x[:, j]
                vj = V[:, j]
                vj = self._fitV(xj, A, vj)
                V[:, j] = vj.reshape(-1, 1)

            if print_loss:
                loss = self._bregman(x, A, V)
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

class LogisticPCA(object):
    def __init__(self, feature):
        self.feature = feature
        self.score = 0
        self.V = 0
        self.theta = 0

    def __len__(self):
        return self.feature

    def _fitScore(self, xi, ai, V):
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
                    gradient[k] += - (xi[j] * V[k, j]) / (1 + exp(xi[j] * thetaj))
            ai -= learning * gradient
            if np.sum((ai - aOld) ** 2) < tol:
                break

        return ai

    def _fitV(self, xj, A, vj):
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
                    gradient[k] += - (xj[i] * A[i, k]) / (1 + exp(xj[i] * thetai))
            vj -= learning * gradient
            if np.sum((vj - vOld) ** 2) < tol:
                break

        return vj

    @staticmethod
    def _bregman(x, A, V):
        n, d = x.shape
        theta = np.dot(A, V)
        loss = 0
        for i in range(n):
            for j in range(d):
                loss += np.log(1 + exp(- x[i, j] * theta[i, j]))

        return loss

    def fit(self, x, max_iter=100,
            tol=1e-4, print_loss=True):
        n, d = x.shape
        x = 2 * x - 1
        feature = self.feature

        # init
        A = np.random.randn(n, feature)
        V = np.random.randn(feature, d)
        theta = np.dot(A, V)
        result = []
        # main loop
        for epoch in range(int(max_iter)):
            thetaOld = theta
            ## update score
            for i in range(n):
                xi = x[i, :]
                ai = A[i, :]
                ai = self._fitScore(xi, ai, V)
                A[i, :] = ai.reshape(-1)
            ## update V
            for j in range(d):
                xj = x[:, j]
                vj = V[:, j]
                vj = self._fitV(xj, A, vj)
                V[:, j] = vj.reshape(-1)

            if print_loss:
                loss = self._bregman(x, A, V)
                result.append(loss)
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
        self.result = result
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

    def plotIter(self):
        plt.plot(self.result)
        plt.show()

class BayesianLogisticPCA(object):
    def __init__(self, feature):
        self.feature = feature
        self.score = 0
        self.V = 0
        self.theta = 0

    def __len__(self):
        return self.feature

    def _fitScore(self, x, theta, V):
        N, D = x.shape
        L = self.feature

        T = np.tanh(theta / 2) / theta
        A = np.zeros((N, L, L))
        for n in range(N):
            for l1 in range(L):
                for l2 in range(L):
                    foo = 0
                    for d in range(D):
                        foo += T[n, d] * V[l1, d] * V[l2, d]
                    A[n, l1, l2] = foo
        B = np.zeros((N, L))
        for n in range(N):
            for l in range(L):
                foo = 0
                for d in range(D):
                    foo += (2 * x[n, d] - 1) * V[l, d]
                B[n, l] = foo
        UList = []
        for n in range(N):
            An = A[n, ...]
            Bn = B[n, :].reshape(-1, 1)
            Un = np.dot(np.linalg.pinv(An), Bn)
            UList.append(Un.reshape(1, -1))
        U = np.vstack(UList)

        return U

    def _fitV(self, x, theta, U):
        N, D = x.shape
        L = self.feature

        T = np.tanh(theta / 2) / theta
        A = np.zeros((D, L, L))
        for d in range(D):
            for l1 in range(L):
                for l2 in range(L):
                    foo = 0
                    for n in range(N):
                        foo += T[n, d] * U[n, l1] * U[n, l2]
                    A[d, l1, l2] = foo
        B = np.zeros((D, L))
        for d in range(D):
            for l in range(L):
                foo = 0
                for n in range(N):
                    foo += (2 * x[n, d] - 1) * U[n, l]
                B[d, l] = foo
        VList = []
        for d in range(D):
            Ad = A[d, ...]
            Bd = B[d, :].reshape(-1, 1)
            Vd = np.dot(np.linalg.pinv(Ad), Bd)
            VList.append(Vd.reshape(-1, 1))
        V = np.hstack(VList)

        return V

    @staticmethod
    def _bregman(x, A, V):
        n, d = x.shape
        truncate = lambda x: min(x, 10) if x > 0 else max(x, -10)
        sigmoid = lambda x: 1 / (1 + exp(-x))
        theta = np.dot(A, V)
        loss = 0
        for i in range(n):
            for j in range(d):
                loss += x[i, j] * np.log(sigmoid(truncate(theta[i, j])))\
                    + (1 - x[i, j]) * np.log(sigmoid(-truncate(theta[i, j])))

        return loss

    def fit(self, x, max_iter=100,
            tol=1e-4, print_loss=True):
        n, d = x.shape
        x = 2 * x - 1
        feature = self.feature

        # init
        U = np.random.randn(n, feature)
        V = np.random.randn(feature, d)
        theta = np.dot(U, V)
        result = []
        # main loop
        for epoch in range(int(max_iter)):
            thetaOld = theta
            ## update score
            U = self._fitScore(x, theta, V)
            ## update theta
            theta = np.dot(U, V)
            ## update V
            V = self._fitV(x, theta, U)
            ## update theta
            theta = np.dot(U, V)

            if print_loss:
                loss = self._bregman(x, U, V)
                result.append(loss)
                print(f'Step {epoch+1}, the loss is {loss:.3f}.')

            if np.sum((thetaOld - theta) ** 2) < tol:
                break

            if epoch == max_iter - 1:
                print(f'Reach max iteration {int(max_iter)}!')

        self.score = U
        self.V = V
        self.theta = theta
        self.prob = self._prob()
        self.label = self._label()
        self.result = result
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

    def plotIter(self):
        plt.plot(self.result)
        plt.show()

if __name__ == '__main__':
    sampleSize = 100
    logit = lambda x: 1 / (1 + np.exp(-x))
    binary = lambda x: max(x - 0.5, 0)/abs(x - 0.5)
    a = np.random.randn(sampleSize) + 0.5
    b = np.random.randn(sampleSize) + 0.5
    x = a
    y = b

    X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    p, q = X.shape
    XProb = logit(X)
    XBin = XProb.copy()
    for i in range(p):
        for j in range(q):
            XBin[i, j] = binary(XBin[i, j])

    model = BayesianLogisticPCA(2)
    model.fit(XBin, max_iter=100)
    model.plotIter()
