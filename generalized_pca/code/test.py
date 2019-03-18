import numpy as np
from numpy.random import RandomState
from scipy.linalg import svd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ipdb
from logistic_pca import LogisticPCA

sampleSize = 4
logit = lambda x: 1 / (1 + np.exp(-x))
binary = lambda x: max(x - 0.5, 0)/abs(x - 0.5)
rdm = RandomState()
a = rdm.randn(sampleSize)
rdm = RandomState()
b = rdm.randn(sampleSize)
x = a + 2 * b
y = a - 2 * b

# natural parameter
theta = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
theta = theta - np.mean(theta, 0)
p, q = theta.shape
# probability via logit trasformation
XProb = logit(theta)
# observed data
XBin = XProb.copy()
for i in range(p):
    for j in range(q):
        XBin[i, j] = binary(XBin[i, j])

model = LogisticPCA(1)
model.fit(XBin, max_iter=1e3, tol=1e-3)
# plot natural parameter space
#plt.scatter(theta[:, 0], theta[:, 1], color='r', label='Original points')
plt.scatter(x, y, color='r', label='Original points')
plt.plot(model.theta[:, 0], model.theta[:, 1], color='b',
         marker='*', label='PCA points')
for i in range(sampleSize):
    plt.plot((x[i], model.theta[i, 0]), (y[i], model.theta[i, 1]), color='g', linestyle='--')
plt.legend()
plt.title("Natural parameter space")
plt.show()

# plot probability space
plt.scatter(XProb[:, 0], XProb[:, 1], color='r', label='Original points')
plt.scatter(model.prob[:, 0], model.prob[:, 1], color='b', label='PCA points')
plt.legend()
plt.title('Probability')
plt.show()
