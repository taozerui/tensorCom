import numpy
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

from ..latent import Latent

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class NMF_Completion(Latent):
    def __init__(self, feature):
        super(NMF_Completion, self).__init__(feature)
        self.U = 0
        self.V = 0

    def __len__(self):
        return super(NMF_Completion, self).__len__()

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

class LogisticPCA_Completion(Latent):
    def __init__(self, feature):
        super(LogisticPCA_Completion, self).__init__(feature)
        self.score = 0
        self.V = 0
        self.theta = 0
        self.prob = 0
        self.label = 0

    def __len__(self):
        return super(LogisticPCA_Completion, self).__len__()

    @staticmethod
    def _bregman(x, A, V, W):
        n, d = x.shape
        theta = W * np.dot(A, V.T)
        foo = np.trace(np.dot(x.T, theta))
        bar = np.sum(np.log(1 + np.exp(theta)))

        return - foo + bar

    def _fitScore(self, X, A, V, W,
                  learning_rate):
        N, p = X.shape
        max_iter = 100
        tol = 1e-2
        for epoch in range(max_iter):
            AOld = A
            theta = np.dot(A, V.T)
            prob = sigmoid(theta)
            gradient = - np.dot(W * X, V) + np.dot(W * prob, V)
            A -= learning_rate * gradient
            #loss = self._bregman(X, A, V, W)
            #print(f'Score step {epoch}, loss {loss}.')
            if np.sum((AOld - A) ** 2) < tol:
                break

        return A

    def _fitV(self, X, A, V, W,
                  learning_rate):
        N, p = X.shape
        max_iter = 100
        tol = 1e-2
        for epoch in range(max_iter):
            VOld = V
            theta = np.dot(A, V.T)
            prob = sigmoid(theta)
            gradient = - np.dot(W.T * X.T, A) + np.dot(W.T * prob.T, A)
            V -= learning_rate * gradient
            if np.sum((VOld - V) ** 2) < tol:
                break

        return V

    def fit(self, x, max_iter=100, learning_rate=0.001,
            tol=1e-4, print_loss=True):
        if type(x) == numpy.ndarray:
            x = csc_matrix(x)

        n, d = x.shape
        #x = 2 * x - 1
        feature = self.feature

        # mask matrix
        W = x.copy()
        W.data = np.ones(len(W.data))
        x = x.toarray()
        W = W.toarray()
        # init
        A = np.random.randn(n, feature)
        V = np.random.randn(d, feature)
        theta = np.dot(A, V.T)
        # main loop
        result = []
        for epoch in range(int(max_iter)):
            thetaOld = theta
            ## update score
            A = self._fitScore(x, A, V, W, learning_rate)
            ## update V
            V = self._fitV(x, A, V, W, learning_rate)

            loss = self._bregman(x, A, V, W)
            result.append(loss)
            if print_loss:
                print(f'Step {epoch+1}, the loss is {loss:.3f}.')

            theta = np.dot(A, V.T)
            if np.sum((thetaOld - theta) ** 2) < tol:
                break

            if epoch == max_iter - 1:
                print(f'Reach max iteration {int(max_iter)}!')

        self.score = A
        self.V = V
        self.theta = theta
        self.prob = self._prob()
        self.label = self._label()
        self.iter = result
        return self

    def _prob(self):
        theta = self.theta

        return sigmoid(theta)

    def _label(self):
        theta = self.theta
        proba = sigmoid(theta)
        label = proba.copy()
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        return label

class LogisticPCA(LogisticPCA_Completion):
    def __init__(self):
        super(LogisticPCA, self).__init__()

    def __len__(self):
        return super(LogisticPCA, self).__len__()

    @staticmethod
    def _bregman(x, A, V, W):
        return super(LogisticPCA)._bregman(x, A, V, W)

    def _fitScore(self, X, A, V, W,
                  learning_rate):
        return super(LogisticPCA, self)._fitScore(X, A, V, W, learning_rate)

    def _fitV(self, X, A, V, W,
                  learning_rate):
        return super(LogisticPCA, self)._fitV(X, A, V, W, learning_rate)

    def fit(self, x, max_iter=100, learning_rate=0.001,
            tol=1e-4, print_loss=True):
        n, d = x.shape
        feature = self.feature

        # mask matrix
        W = np.ones((n, d))
        # init
        A = np.random.randn(n, feature)
        V = np.random.randn(d, feature)
        theta = np.dot(A, V.T)
        # main loop
        result = []
        for epoch in range(int(max_iter)):
            thetaOld = theta
            ## update score
            A = self._fitScore(x, A, V, W, learning_rate)
            ## update V
            V = self._fitV(x, A, V, W, learning_rate)

            loss = self._bregman(x, A, V, W)
            result.append(loss)
            if print_loss:
                print(f'Step {epoch+1}, the loss is {loss:.3f}.')

            theta = np.dot(A, V.T)
            if np.sum((thetaOld - theta) ** 2) < tol:
                break

            if epoch == max_iter - 1:
                print(f'Reach max iteration {int(max_iter)}!')

        self.score = A
        self.V = V
        self.theta = theta
        self.prob = self._prob()
        self.label = self._label()
        self.iter = result
        return self

    def _prob(self):
        return super(LogisticPCA, self)._prob()

    def _label(self):
        return super(LogisticPCA, self)._label()
