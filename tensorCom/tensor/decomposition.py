import sparse
import numpy as np
import tensorBase as T
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

class Latent(object):
    def __init__(self, ranks):
        self.ranks = ranks
        self.Xhat = 0
        self.iter = 0

    def __len__(self):
        return self.ranks

    def plotIter(self):
        plt.plot(self.iter)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Iteration')
        plt.show()

class Tucker(Latent):
    def __init__(self, ranks):
        super(Tucker, self).__init__(ranks)
        self.core = 0
        self.factors = 0

    def __len__(self):
        return super(Tucker, self).__len__()

    def fit(self, x, learning_rate=0.01,
            max_iter=100, tol=1e-3, print_loss=False):

        ranks = self.ranks
        order = len(x.shape)
        if len(ranks) != order:
            raise ValueError('Wrong number of ranks!')

        # init
        factors = []
        for i in range(order):
            Ai = np.random.rand(x.shape[i], ranks[i])
            factors.append(Ai)
        G = np.random.rand(*ranks)
        # main loop
        loss = 0
        iter = []
        for epoch in range(int(max_iter)):
            lossOld = loss
            # update factors
            for i in range(order):
                foo = T.modeProductList(G, factors) - x
                foo = T.unfold(foo, i)
                bar = T.modeProductList(G, factors, skip=i)
                bar = T.unfold(bar, i)
                gradient = np.dot(foo, bar.T)
                Ai = factors[i] - learning_rate * gradient
                factors[i] = Ai
            # updata core
            foo = T.modeProductList(G, factors) - x
            factorsT = [ll.T for ll in factors]
            gradient = T.modeProductList(foo, factorsT)
            G = G - learning_rate * gradient
            xhat = T.modeProductList(G, factors)
            loss = np.sum((x - xhat) ** 2)
            iter.append(loss)
            # print loss
            if print_loss:
                print(f'Step {epoch+1}, the loss is {loss:.3f}.')
            # convergence condition
            if np.abs(lossOld - loss) < tol:
                break
            if epoch == max_iter - 1:
                print(f'Reach max iteration {max_iter}!')

        self.core = G
        self.factors = factors
        self.iter = iter
        self.Xhat = xhat
        return self

    def plotIter(self):
        super(Tucker, self).plotIter()

class LogisticCP(Latent):
    def __init__(self, ranks):
        super(LogisticCP, self).__init__(ranks)
        self.core = 0
        self.factors = 0

    def __len__(self):
        return super(LogisticCP, self).__len__()


    @staticmethod
    def _updateFactor(x, factors, mode, learning_rate):
        xk = T.unfold(x, mode)
        y = T.vec(xk)
        foo = factors.copy()
        foo.pop(mode)
        AInvek = T.khatri_rao(foo, is_reverse=True)
        predictor = np.kron(AInvek, np.eye(x.shape[mode]))
        n, p = predictor.shape
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # init
        Ak = factors[mode]
        shapek = Ak.shape
        beta = T.vec(Ak)
        theta = np.dot(predictor, beta)
        for i in range(500):
            betaOld = beta.copy()
            prob = sigmoid(theta).reshape(n, 1)
            gradient = - np.dot(predictor.T, (y - prob))
            beta -= learning_rate * gradient
            if np.sum((beta - betaOld) ** 2) / len(beta) < 1e-2:
               break
            theta = np.dot(predictor, beta)

        return beta.reshape(*shapek)

    @staticmethod
    def _crossEntropy(x, theta):
        q = 2 * x - 1
        theta = q * theta
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        loglike = np.sum(np.log(sigmoid(theta)))

        return - loglike

    def fit(self, x, learning_rate=0.001,
            max_iter=100, tol=1e-3, print_loss=False):
        ranks = self.ranks
        order = len(x.shape)

        # init
        factors = []
        for i in range(order):
            Ai = np.random.randn(x.shape[i], ranks)
            factors.append(Ai)
        # main loop
        result = []
        loss = 0
        for epoch in range(int(max_iter)):
            factorsOld = factors.copy()
            lossOld = loss
            # update each factor
            for k in range(order):
                Ak = self._updateFactor(x, factors, k, learning_rate)
                factors[k] = Ak
            # line search
            # pass this step
            # normalization
            lmbd = np.ones(ranks)
            for k in range(order):
                Ak = factors[k]
                AK = factors[-1]
                lmbd *= np.sqrt(np.sum(Ak ** 2, axis=0))
                Ak /= lmbd
                AK *= lmbd
                factors[k] = Ak
                factors[-1] = AK
            theta = T.unCP(factors)
            loss = self._crossEntropy(x, theta)
            result.append(loss)
            if print_loss:
                print(f'Step {epoch}, the loss is {loss:.3f}.')
            if np.abs(lossOld - loss) < tol:
                break
            if epoch == max_iter - 1:
                print(f'Reach max iteration {int(max_iter)}!')

        self.factors = factors
        self.iter = result
        return self

    def plotIter(self):
        super(LogisticCP, self).plotIter()

class GeneralizedTucker(Latent):
    def __init__(self, ranks):
        super(GeneralizedTucker, self).__init__(ranks)
        self.core = 0
        self.factors = 0

    def __len__(self):
        return super(GeneralizedTucker, self).__len__()

    def fit(self, x, learning_rate=0.01,
            max_iter=100, tol=1e-3, print_loss=False):
        pass

    def plotIter(self):
        super(GeneralizedTucker, self).plotIter()

if __name__ == '__main__':
    x = np.random.rand(5, 6, 7)
    x[x > 0.5] = 1
    x[x <= 0.5] = 0
    model = LogisticCP(3)
    model.fit(x, learning_rate=0.001, max_iter=2000, print_loss=True)
    model.plotIter()
