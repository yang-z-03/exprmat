
import numpy as np
import anndata as ad
from exprmat.ansi import error


def ensure_array(a):
    ''' If a is a matrix, turn it into an array. '''
    if isinstance(a, np.matrix): return a.A
    else: return a


def choose_layer(adata, use_raw = False, layer = None):
    is_layer = layer is not None
    if is_layer: 
        if layer == 'X': return adata.X
        else: return adata.layers[layer]
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