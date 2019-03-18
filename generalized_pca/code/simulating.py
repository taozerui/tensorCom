import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ipdb
from logistic_pca import LogisticPCA

def normal(sampleSize=10):
    a = np.random.randn(sampleSize)
    b = np.random.randn(sampleSize)
    x = a + 2 * b
    y = a - 2 * b

    X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    X = X - np.mean(X, 0)
    features = 1
    # pca
    U, D, VT = svd(X)
    V = VT.T
    Vq = V[:, 0:features].reshape(-1, 1)
    Hq = np.dot(Vq, Vq.T)
    theta = np.dot(X, Hq)

    plt.figure(figsize=(5, 5))
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X[:, 0], X[:, 1], color='r')
    plt.plot(theta[:, 0], theta[:, 1], color='b', marker='*')
    for i in range(sampleSize):
        plt.plot((X[i, 0], theta[i, 0]), (X[i, 1], theta[i, 1]), color='g', linestyle='--')
    plt.show()

def bernoulli(sampleSize=10):
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

    return X, XProb, XBin

def main(sampleSize=10):
    X, XProb, XBin = bernoulli(sampleSize)

    features = 1
    # pca
    U, D, VT = svd(XBin)
    V = VT.T
    Vq = V[:, 0:features].reshape(-1, 1)
    Hq = np.dot(Vq, Vq.T)
    theta = np.dot(XBin, Hq)
    plt.figure()
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)
    plt.scatter(XProb[:, 0], XProb[:, 1], color='r')
    plt.plot(theta[:, 0], theta[:, 1], color='b', marker='*')
    for i in range(sampleSize):
        plt.plot((XProb[i, 0], theta[i, 0]), (XProb[i, 1], theta[i, 1]), color='g', linestyle='--')
    plt.show()

    # logistic pca
    model = LogisticPCA(features)
    model.fit(XBin, max_iter=500, tol=1e-3)
    ipdb.set_trace()
    plt.figure()
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)
    plt.scatter(XProb[:, 0], XProb[:, 1], color='r')
    plt.scatter(model.label()[:, 0], model.label()[:, 1], color='b')
    plt.show()

if __name__ == '__main__':
    normal(10)
