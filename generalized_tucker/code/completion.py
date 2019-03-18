import pickle
import ipdb
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
        plt.plot(iter)
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
        # mask tensor
        W = x.copy()
        W.data = np.ones(len(W.data))
        W = W.todense()
        x = x.todense()
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
                foo = W * foo
                foo = T.unfold(foo, i)
                bar = T.modeProductList(G, factors, skip=i)
                bar = T.unfold(bar, i)
                gradient = np.dot(foo, bar.T)
                Ai = factors[i] - learning_rate * gradient
                factors[i] = Ai
            # updata core
            foo = T.modeProductList(G, factors) - x
            foo = W * foo
            factorsT = [ll.T for ll in factors]
            gradient = T.modeProductList(foo, factorsT)
            G = G - learning_rate * gradient
            xhat = T.modeProductList(G, factors)
            loss = np.sum((W * (x - xhat)) ** 2)
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
