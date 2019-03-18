import numpy as np

def _norm(x, axis=0):
    n, p = x.shape
    meanX = np.mean(x, axis=axis)

    return x - meanX

def power(x, r, init='random',
          max_iter=100, tol=1e-3):
    '''
    input
        - x, the original matrix
        - r, the dimension of latent features
        - init, 'random' or 'zero'
        - max_iter, maximum iteration steps
        - tol, convergence condition
    output
        - (A, V), latent features
    '''
    n, p = x.shape
    # init
    if init == 'random':
        A = np.random.randn(n, r)
        V = np.random.randn(r, p)
    elif init == 'zero':
        A = np.zeros((n, r))
        V = np.zeros((r, p))
    else:
        raise ValueError('Wrong initialization method!')

    for i in range(n):
        for c in range(r):
            pass

    pass
