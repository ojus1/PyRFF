#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : October 11, 2020
# Modified by Surya Kant Sahu <surya.oju@gmail.com>
# Removed Scipy Dependency, Works with Sequential Features

##################################################### SOURCE START #####################################################


import functools
import numpy as np

# Fix random seed used in this script.


def seed(seed):

    # Now it is enough to fix the random seed of Numpy.
    np.random.seed(seed)


# Generate random matrix for random Fourier features.
def get_rff_matrix(dim_in, dim_out, std):

    return std * np.random.randn(dim_in, dim_out)


# Generate random matrix for orthogonal random features.
def get_orf_matrix(dim_in, dim_out, std):

    # Initialize matrix W.
    W = None

    for _ in range(dim_out // dim_in + 1):
        s = np.sqrt(np.random.chisquare(df=dim_in, size=(dim_in, )))
        Q = np.linalg.qr(np.random.randn(dim_in, dim_in))[0]
        V = std * np.dot(np.diag(s), Q)
        W = V if W is None else np.concatenate([W, V], axis=1)

    # Trim unnecessary part.
    return W[:dim_in, :dim_out]


# This function returns a function which generate RFF/ORF matrix.
# The usage of the returned value of this function are:
# f(dim_input:int) -> np.array with shape (dim_input, dim_kernel)
def get_matrix_generator(rand_mat_type, std, dim_kernel):

    if rand_mat_type == "rff":
        return functools.partial(get_rff_matrix, std=std, dim_out=dim_kernel)
    elif rand_mat_type == "orf":
        return functools.partial(get_orf_matrix, std=std, dim_out=dim_kernel)
    else:
        raise RuntimeError(
            "matrix_generator: 'rand_mat_type' must be 'rff' or 'orf'.")


def _pad(vec, max_length):
    '''
    vec: List of numpy arrays with shapes (*, N), where * can be varying.
    max_length: int denoting the maximum padded length.
    returns: numpy array of shape (max_length, N)
    '''
    ret = np.zeros((len(vec), max_length, vec[0].shape[-1]))
    for i, v in enumerate(vec):
        ind = min(max_length, v.shape[0])
        ret[i, :ind, :] = v[:ind, :]
    return ret

# Base class of the following RFF/ORF related classes.


class Base:

    # Constructor. Create random matrix generator and random matrix instance.
    # NOTE: If 'W' is None then the appropriate matrix will be set just before the training.
    def __init__(self, rand_mat_type, dim_kernel, std_kernel, W, max_length=None):
        self.dim = dim_kernel
        self.s_k = std_kernel
        self.mat = get_matrix_generator(rand_mat_type, std_kernel, dim_kernel)
        self.W = W
        self.max_length = max_length

    # Apply random matrix to the given input vectors 'X' and create feature vectors.
    # NOTE: This function can manipulate multiple random matrix. If argument 'index'
    # is given, then use self.W[index] as a random matrix, otherwise use self.W itself.
    def conv(self, X, index=None):
        W = self.W if index is None else self.W[index]
        if isinstance(X, list):
            assert(self.max_length is not None)
            X = _pad(X, self.max_length).reshape((len(X), -1))
        ts = X @ W
        return np.bmat([np.cos(ts), np.sin(ts)])

    # Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).
    # NOTE: This function can manipulate multiple random matrix. If argument 'dim_in'
    # is a list/tuple of integers, then generate multiple random matrixes.

    def set_weight(self, dim_in):
        if hasattr(dim_in, "__iter__"):
            self.W = self.mat(np.prod(dim_in))
        else:
            self.W = self.mat(dim_in)


def get_features_sequential(X, random_seed, *args, **kwargs):
    '''
    Returns Fourier Features for List of vectors X
    Args:
        random_seed: Seed for random matrix
        matrix_type: "orf" or "rrf"
        num_fourier_features: int, Returns 2 * num_fourier_features output tensor  
        std: float, Standard deviation for random matrix
        max_length: int, Pad to this length for sequential features 
    '''
    seed(random_seed)
    base = Base(*args, None, **kwargs)
    base.set_weight((kwargs['max_length'], X[0].shape[-1]))
    return base.conv(X)


def get_features(X, random_seed, *args, **kwargs):
    '''
    Returns Fourier Features for List of vectors X
    Args:
        random_seed: Seed for random matrix
        matrix_type: "orf" or "rrf"
        num_fourier_features: int, Returns 2 * num_fourier_features output tensor  
        std: float, Standard deviation for random matrix
    '''
    seed(random_seed)
    base = Base(*args, None, None, **kwargs)
    base.set_weight(X[0].shape[-1])
    return base.conv(X)
