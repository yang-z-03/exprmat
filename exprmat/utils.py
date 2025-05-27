
import numpy as np
import anndata as ad
import scipy.sparse as sp
from functools import singledispatch
from numba import njit

import numba
from exprmat.ansi import error, warning, info


plotting_styles = {
    'font_name': ['Helvetica Neue LT Std', 'Helvetica', 'Arial', 'Ubuntu', 'Verdana'],
    'backend': 'TkAgg'
}

def setup_styles(font_name = 'Helvetica Neue LT Std', backend = 'TkAgg'):

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # mpl.use(backend)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_name
    plt.rcParams["ytick.labelright"] = False
    plt.rcParams["ytick.labelleft"] = True

    np.set_printoptions(precision = 3, floatmode = 'fixed', suppress = True)
    pass


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


def translate_variables(adata, gene_list, layer = 'X'):
    
    ensembls = adata.var['ensembl'].tolist()
    index = adata.var_names.tolist()
    names = adata.var['gene'].tolist()

    target = []
    for x in gene_list:
        if x in ensembls:
            target.append(index[ensembls.index(x)])
        elif x in names:
            target.append(index[names.index(x)])
        elif x in index:
            target.append(x)
        elif ('rna:' + x) in index:
            target.append('rna:' + x)
        else: warning(f'unable to find gene `{x}` in variables.')
    
    return target


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
    else: error('not implemented.')


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
    else: error('not implemented.')


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
    else: error('not implemented.')


def mdata_to_adata_2(
    mdata,
    x_mod, y_mod,
    x_layer = None, y_layer = None,
    x_use_raw = False, y_use_raw = False,
    x_transform = None,
    y_transform = None,
    verbose = True
):
    """
    Convert a MultiData object to an AnnData object. Returns an AnnData object with the 
    two modalities concatenated. Information related to observations (obs, obsp, obsm) 
    and `.uns` are copied from the original MuData object.
    """

    if x_mod is None or y_mod is None:
        error("both `x_mod` and `y_mod` must be provided.")

    xdata = handle_modality(mdata, x_mod, x_use_raw, x_layer, x_transform, verbose)
    ydata = handle_modality(mdata, y_mod, y_use_raw, y_layer, y_transform, verbose)

    adata = ad.concat([xdata, ydata], axis = 1, label = 'modality')

    adata.obs = mdata.obs.copy()
    adata.obsp = mdata.obsp.copy()
    adata.obsm = mdata.obsm.copy()
    adata.uns = mdata.uns.copy()

    return adata


def handle_modality(mdata, mod, use_raw, layer, transform, verbose):
    
    if mod not in mdata.mod.keys():
        error(f'`{mod}` is not in the mdata!')

    md = mdata.mod[mod].copy()
    if use_raw: md = md.raw.to_adata()
    else: md.X = choose_layer(md, use_raw = use_raw, layer = layer)

    if transform:
        if verbose: info(f'transforming {mod} using {transform.__name__}')
        md.X = transform(md.X)
    
    return md