import numpy as np


def pca(features, dim_remain=2):
    # singular value decomposition factorises your data matrix such that:
    #
    #   features = U * S * V.T     (where '*' is matrix multiplication)
    #   features * V = U * S
    #
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    #
    # * S is diagonal matrix containing the singular values of features - these
    #   values squared divided by the number of observations will give the
    #   variance explained by each PC.
    #
    # * if features is considered to be an (observations, features) matrix, the PCs
    #   themselves would correspond to the rows of S^(1/2)*V.T. if features is
    #   (features, observations) then the PCs would be the columns of
    #   U*S^(1/2).
    #
    # * since U and V both contain orthonormal vectors, U*V.T is equivalent
    #   to a whitened version of features.

    U, s, Vt = np.linalg.svd(features, full_matrices=False)
    V = Vt.T
    S = np.diag(s)

    # PCs are already sorted by descending order
    # of the singular values (i.e. by the
    # proportion of total variance they explain)
    '''
    # Method 1 reconstruction:
    # if we use all of the PCs we can reconstruct the original signal perfectly.

    Mhat1 = np.dot(U, np.dot(S, V.T))
    print('Using all PCs, MSE = %.6G' %(np.mean((features.values - Mhat)**2)))

    # Method 2 weak reconstruction:
    # if we use only the first few PCs the reconstruction is less accurate.
    # the dimention is remained the same sa before, but some information is
    # lost in this reconstruction process.
    Mhat2 = np.dot(U[:, :dim_remain], np.dot(S[:dim_remain, :dim_remain], V[:,:dim_remain].T))
    print('Not a Number is located there: ', np.where(np.isnan(features.values) == True))
    print('Using first few PCs, MSE = %.6G' %(np.mean((features.values - Mhat2)**2)))
    '''
    # Method 3 dimention reduction:
    # if we use only the first few PCs the reconstruction is less accurate,
    # the dimension is also recuded to (or to say projected on) into another
    # low dimenional space.
    Mhat = np.dot(U[:, :dim_remain], S[:dim_remain, :dim_remain])

    return Mhat, V[:, :dim_remain]
