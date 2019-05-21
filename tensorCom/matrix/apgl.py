#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.linalg import svd
from numpy.random import RandomState
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


def svt(x, coef):
    """SVT algorithm
    Parameters
    ----------
    x : numpy.ndarray
        The matrix to be shrinked.
    coef : float
        Shrinkage coefficient

    Returns
    -------
    xHat : numpy.ndarray
        Shrinked matrix
    """
    n1, n2 = x.shape
    U, D, VT = svd(x)
    d = len(D)
    DTrun = np.zeros((n1, n2))
    for i in range(d):
        DTrun[i, i] = max(D[i] - coef, 0)

    return U @ DTrun @ VT


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _gradient(x, mask, theta):
    g = mask * x - mask * _sigmoid(theta)

    return -2 * g


def _cross_entropy(x, mask, theta):
    f1 = np.sum(mask * x * theta)
    f2 = np.sum(mask * np.log(1 + np.exp(theta)))

    return -2 * (f1 - f2)


def _bound(x, mask, theta, thetaOld, lr):
    g1 = _cross_entropy(x, mask, thetaOld)
    g2 = np.sum(_gradient(x, mask, thetaOld) * (theta - thetaOld))
    g3 = np.sum((theta - thetaOld) ** 2) / (2 * lr)

    return g1 + g2 + g3


def _fit_proximal(x, mask, lmbd,
                  max_iter=100, L=1e-3, beta=0.5,
                  tol=1e-3, print_loss=False):
    """ Proximal based solver for nuclear norm optimization problems.
    Input
    x : np.ndarray
        partially observed matrix.
    mask: np.ndarray
        mask matrix
    lmbd : float
        penalization coef.
    L : float
        learning rate, default 1e-3.
    beta : float in (0, 1)
        decay coef default 0.5.
    """
    # init
    n1, n2 = x.shape
    rdm = RandomState(123)
    theta = rdm.randn(n1, n2)  # natural parameter

    # main loop
    loss = _cross_entropy(x, mask, theta) + lmbd * \
        np.linalg.norm(theta, ord='nuc')
    iteration = []
    for i in range(int(max_iter)):
        if print_loss:
            print(f'Epoch {i}, loss {loss:.3f}')
        iteration.append(loss)
        lossOld = loss
        for _ in range(50):
            S = theta - L * _gradient(x, mask, theta)
            thetaNew = svt(S, lmbd * L)
            ce = _cross_entropy(x, mask, thetaNew)
            if ce < _bound(x, mask, thetaNew, theta, L):
                break
            else:
                L = beta * L
        theta = thetaNew
        loss = ce + lmbd * np.linalg.norm(theta, ord='nuc')
        if i == max_iter - 1:
            print(f'Reach max iteration {i+1}')
        if np.abs(lossOld - loss) < tol:
            break

    return theta, np.array(iteration)


def _fit_apgl(x, mask, lmbd,
              max_iter=100, L=1e-3, beta=0.5,
              tol=1e-3, print_loss=False):
    """ Proximal based solver for nuclear norm optimization problems.
    Input
    x : np.ndarray
        partially observed matrix.
    mask: np.ndarray
        mask matrix.
    lmbd : float
        penalization coef.
    L : float
        learning rate, default 1e-3.
    beta : float in (0, 1)
        decay coef default 0.5.
    """
    # init
    n1, n2 = x.shape
    rdm = RandomState(123)
    theta = rdm.randn(n1, n2)  # natural parameter
    thetaOld = theta
    alpha = 1
    alphaOld = 0

    # main loop
    loss = _cross_entropy(x, mask, theta) + lmbd * \
        np.linalg.norm(theta, ord='nuc')
    iteration = []
    for i in range(int(max_iter)):
        if print_loss:
            print(f'Epoch {i}, loss {loss:.3f}')
        iteration.append(loss)
        lossOld = loss
        # nesterov extropolation
        A = theta + (alphaOld - 1) / alpha * (theta - thetaOld)
        for _ in range(50):
            S = A - L * _gradient(x, mask, A)
            thetaNew = svt(S, lmbd * L)
            ce = _cross_entropy(x, mask, thetaNew)
            if ce < _bound(x, mask, thetaNew, theta, L):
                break
            else:
                L = beta * L
        thetaOld = theta
        theta = thetaNew
        alphaOld = alpha
        alpha = (1 + np.sqrt(4 + alpha ** 2)) / 2
        loss = ce + lmbd * np.linalg.norm(theta, ord='nuc')
        if i == max_iter - 1:
            print(f'Reach max iteration {i+1}')
        if np.abs(lossOld - loss) < tol:
            break

    return theta, np.array(iteration)


class BinaryMC(object):
    """Binary matrix completion, using proximal based algorithm.

    """

    def __init__(self, lmbd):
        self.lmbd = lmbd

    def fit(self, x, mask, solver='apgl', lr=1e-3, beta=0.5,
            max_iter=100, tol=1e-3, print_loss=False):
        if solver == 'proximal':
            theta, iteration = _fit_proximal(x, mask, self.lmbd,
                                             max_iter=max_iter, L=lr,
                                             beta=beta, tol=tol,
                                             print_loss=print_loss)
        elif solver == 'apgl':
            theta, iteration = _fit_apgl(x, mask, self.lmbd,
                                         max_iter=max_iter, L=lr,
                                         beta=beta, tol=tol,
                                         print_loss=print_loss)
        else:
            raise Exception('Wrong solver!')
        proba = _sigmoid(theta)
        label = proba.copy()
        label[proba > 0.5] = 1
        label[proba <= 0.5] = 0

        self.theta = theta
        self.proba = proba
        self.label = label
        self.iteration = iteration

        return self

    def plotIter(self):
        plt.figure()
        plt.plot(self.iteration)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()


if __name__ == '__main__':
    N1 = 1000
    N2 = 1200
    R = 5
    # generate data
    rdm = np.random.RandomState(123)
    U = rdm.randn(N1, R)
    V = rdm.randn(N2, R)
    theta = np.dot(U, V.T)
    # mask
    obsRate = 0.2
    mask = rdm.binomial(1, obsRate * np.ones((N1, N2)))
    # generate label
    prob = _sigmoid(theta)
    label = rdm.binomial(1, prob)
    labelTrain = label.copy()
    labelTrain[mask == 0] = 0

    # benchmark
    is_benchmark = False
    if is_benchmark:
        from fancyimpute import SoftImpute
        labelTrainBen = label.copy()
        labelTrainBen = np.array(label, dtype=float)
        labelTrainBen[mask == 0] = np.nan
        labelBench = SoftImpute().fit_transform(labelTrainBen)
        labelBench[labelBench > 0.5] = 1
        labelBench[labelBench <= 0.5] = 0
        maskInv = 1 - mask
        errorBench = np.sum(np.abs(maskInv * (labelBench - label))
                            ) / np.sum(maskInv)
        print(f'The benchmark error by SoftImpute is {errorBench:.3f}')

    # train
    ls = np.r_[30]
    for l in ls:
        model1 = BinaryMC(l)
        model1.fit(labelTrain, mask, solver='proximal',
                   max_iter=50, lr=0.01, print_loss=True)
        # model.plotIter()
        model2 = BinaryMC(l)
        model2.fit(labelTrain, mask, solver='apgl',
                   max_iter=50, lr=0.01, print_loss=True)
        plt.plot(model1.iteration, color='r', label='proximal')
        plt.plot(model2.iteration, color='b', label='apgl')
        plt.title('Iteration procedure')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # model.plotIter()
        # test
        maskInv = 1 - mask
        error1 = np.sum(np.abs(maskInv * (model1.label - label))
                        ) / np.sum(maskInv)
        error2 = np.sum(np.abs(maskInv * (model2.label - label))
                        ) / np.sum(maskInv)
        print('The mis-classification error is', end=' ')
        print(f'| \'proximal\' {error1:.3f} | \'apgl\' {error2:.3f}.')
