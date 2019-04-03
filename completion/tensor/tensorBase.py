import numpy as np
from scipy.sparse import csr_matrix

def vec(x):
    return x.reshape(-1, 1, order='F')

def unVec(x, p, q):
    return x.reshape(p, q, order='F')

def fold(x, mode, shape):
    fullShape = list(shape)
    modeDim = fullShape.pop(mode)
    fullShape.insert(0, modeDim)
    return np.moveaxis(np.reshape(x, fullShape, order='F'), 0, mode)

def unfold(x, mode):
    out = np.reshape(np.moveaxis(x, mode, 0), (x.shape[mode], -1), order='F')
    return out

def khatri_rao(matrix_list, skip_matrix=None, is_reverse=False):
    cache = matrix_list.copy()
    n, r = cache[0].shape
    for matrix in cache:
        if matrix.shape[1] != r:
            raise ValueError("Dimension error!")

    if skip_matrix != None:
        cache.pop(skip_matrix)

    if is_reverse:
        cache.reverse()

    columns = []
    for i in range(r):
        column_i = 1
        for matrix in cache:
            column_i = np.kron(column_i, matrix[:, i])

        columns.append(vec(column_i))

    return np.hstack(columns)

def partialUnfold(tensor, mode=0, skip_begin=1, skip_end=0, ravel_tensors=False):
    if ravel_tensors:
        new_shape = [-1]
    else:
        new_shape = [tensor.shape[mode + skip_begin], -1]

    if skip_begin:
        new_shape = [tensor.shape[i] for i in range(skip_begin)] + new_shape

    if skip_end:
        new_shape += [tensor.shape[-i] for i in range(skip_end)]

    out = np.reshape(np.moveaxis(tensor, mode+skip_begin, skip_begin), new_shape, order='F')
    return out

def matrixDot(A, B):
    return np.dot(vec(A).T, vec(B))

def unCP(Bds):
    fullShape = []
    for i in Bds:
        fullShape.append(i.shape[0])
    factors = list(Bds)
    factors.pop(0)
    factors.reverse()
    x1 = np.dot(Bds[0], khatri_rao(factors).T)

    return fold(x1, 0, fullShape)

def modeProduct(tensor, matrix, mode):
    if tensor.shape[mode] != matrix.shape[1]:
        raise ValueError('Dimension error!')

    outFold = np.dot(matrix, unfold(tensor, mode))
    fullShape = list(tensor.shape)
    fullShape[mode] = matrix.shape[0]

    return fold(outFold, mode, fullShape)

def modeProductList(tensor, matrix_list, skip=None):
    out = tensor
    for k, matrix in enumerate(matrix_list):
        if skip == None:
            out = modeProduct(out, matrix, k)
        else:
            if k == skip:
                pass
            else:
                out = modeProduct(out, matrix, k)

    return out

def indexVector(i, j):
    """
    generate an index vector of length i, and with the j-th non-zero element
    """
    x = np.zeros(i)
    x[j] = 1
    return x

def commutation(m, n):
    """
    calculate the commutation matrix for A of size m times n
    """
    rowIndex = np.array([])
    colIndex = np.array([])
    for i in range(m):
        for j in range(n):
            rowIndex = np.append(rowIndex, i * n + j)
            colIndex = np.append(colIndex, j * m + i)
    data = np.ones(m * n)
    Kmn = csr_matrix((data, (rowIndex, colIndex)), shape=(m * n, m * n))

    return Kmn

def commuKhatriMatrix(A, p):
    A = A.get()
    m, n = A.shape
    col = np.array([])
    data = np.array([])
    for n_i in range(n):
        for m_i in range(m):
            for p_i in range(p):
                col = np.append(col, n_i * p + p_i)
                data = np.append(data, A[m_i, n_i])

    row = np.linspace(0, m * p * n - 1, m * p * n)
    return csr_matrix((data, (row, col)), shape=(m * n * p, p * n))
