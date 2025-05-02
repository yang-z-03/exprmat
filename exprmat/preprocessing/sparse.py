'''
Utility functions to manipulate sparse matrix.
'''

import scipy
import numpy as np


def sparse_var(E, axis = 0):
    '''
    Variance across the specified axis.
    Axis 0 indicates variance by column, and axis 1 by row.
    '''

    mean_gene = E.mean(axis = axis).A.squeeze()
    tmp = E.copy()
    tmp.data **= 2
    return tmp.mean(axis = axis).A.squeeze() - mean_gene ** 2


def sparse_multiply_row(E, a):
    ''' Multiply each row of E by a scalar '''

    nrow = E.shape[0]
    w = scipy.sparse.lil_matrix((nrow, nrow))
    w.setdiag(a)
    return w * E


def sparse_zscore_column(E, gene_mean = None, gene_stdev = None):
    ''' Scale the z-score for columns of E '''

    if gene_mean is None:
        gene_mean = E.mean(0)
    if gene_stdev is None:
        gene_stdev = np.sqrt(sparse_var(E))
    return sparse_multiply_row((E - gene_mean).T, 1/gene_stdev).T

