
import numpy as np
import anndata as ad
import scipy.sparse as sp
from functools import singledispatch
from numba import njit

import numba
import numpy as np
from exprmat.ansi import error


plotting_styles = {
    'font_name': ['Arial', 'Helvetica', 'Ubuntu', 'Verdana'],
    'backend': 'TkAgg'
}

def setup_styles(font_name = 'Arial', backend = 'TkAgg'):

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # mpl.use(backend)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_name


def ensure_array(a):
    ''' If a is a matrix, turn it into an array. '''
    if isinstance(a, np.matrix): return a.A
    else: return a


def choose_layer(adata, use_raw = False, layer = None):
    is_layer = layer is not None
    if is_layer: 
        if layer == 'X': return adata.X
        elif layer in adata.layers.keys(): return adata.layers[layer]
        else: error(f'layer `{layer}` does not present in the annotated data.')
    elif use_raw: return adata.raw.X
    else: return adata.X


def choose_representation(
    adata: ad.AnnData, *, use_rep = 'pca', n_pcs = None, silent = False,
):
    if use_rep in adata.obsm and n_pcs is not None:
        if n_pcs > adata.obsm[use_rep].shape[1]:
            error(f'{use_rep} does not have enough dimensions.')
        X = adata.obsm[use_rep][:, :n_pcs]

    elif use_rep in adata.obsm and n_pcs is None:
        X = adata.obsm[use_rep]

    elif use_rep == 'X':
        X = adata.X

    else: error(f'{use_rep} does not exist is the obsm slots.')
    return X


def align(subset, parent):
    return [None if not x in parent else parent.index(x) for x in subset]


def find_variable(adata, gene_name, layer = 'X'):

    from scipy.sparse import issparse

    if gene_name in adata.obs.columns:
        vec = adata.obs[gene_name].tolist()

    elif gene_name in adata.var_names:
        X = adata[:, gene_name].X if layer == 'X' else adata[:, gene_name].layers[layer]
        if issparse(X): vec = X.toarray().reshape(-1)
        else: vec = X.reshape(-1)

    # try some conventions
    elif 'gene' in adata.var.keys() and \
        gene_name in adata.var['gene'].tolist():
        genes = adata.var['gene'].tolist()
        X = adata.X[:, genes.index(gene_name)] if layer == 'X' \
            else adata[:, genes.index(gene_name)].layers[layer]
        if issparse(X): vec = X.toarray().reshape(-1)
        else: vec = X.reshape(-1)

     # try some conventions
    elif 'ensembl' in adata.var.keys() and \
        gene_name in adata.var['ensembl'].tolist():
        genes = adata.var['ensembl'].tolist()
        X = adata.X[:, genes.index(gene_name)] if layer == 'X' \
            else adata[:, genes.index(gene_name)].layers[layer]
        
        if issparse(X): vec = X.toarray().reshape(-1)
        else: vec = X.reshape(-1)

    else: error(f'unable to find gene `{gene_name}` in metadata or variables.')
    return vec


@singledispatch
def is_constant(a, axis = None):
    """ Check whether values in array are constant. """
    raise NotImplementedError()


@is_constant.register(np.ndarray)
def _(a, axis = None):
    # should eventually support nd, not now.
    if axis is None: return bool((a == a.flat[0]).all())
    if axis == 0: return _is_constant_rows(a.T)
    elif axis == 1: return _is_constant_rows(a)


def _is_constant_rows(a):
    b = np.broadcast_to(a[:, 0][:, np.newaxis], a.shape)
    return (a == b).all(axis = 1)


@is_constant.register(sp.csr_matrix)
def _(a, axis = None):
    if axis is None:
        if len(a.data) == np.multiply(*a.shape):
            return is_constant(a.data)
        else: return (a.data == 0).all()
    if axis == 1: return _is_constant_csr_rows(a.data, a.indptr, a.shape)
    elif axis == 0:
        a = a.T.tocsr()
        return _is_constant_csr_rows(a.data, a.indptr, a.shape)


@njit
def _is_constant_csr_rows(data, indptr, shape,):
    n = len(indptr) - 1
    result = np.ones(n, dtype=np.bool_)
    for i in numba.prange(n):
        start = indptr[i]
        stop = indptr[i + 1]
        val = data[start] if stop - start == shape[1] else 0
        for j in range(start, stop):
            if data[j] != val:
                result[i] = False
                break
    return result


@is_constant.register(sp.csc_matrix)
def _(a, axis = None):
    if axis is None:
        if len(a.data) == np.multiply(*a.shape): return is_constant(a.data)
        else: return (a.data == 0).all()
    if axis == 0: return _is_constant_csr_rows(a.data, a.indptr, a.shape[::-1])
    elif axis == 1:
        a = a.T.tocsc()
        return _is_constant_csr_rows(a.data, a.indptr, a.shape[::-1])
