import matplotlib.pyplot as plt

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
