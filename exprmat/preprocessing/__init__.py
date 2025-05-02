
import numpy as np

from exprmat.preprocessing.linear import normalize_linear_methods
from exprmat.preprocessing.normal import log1p
from exprmat.preprocessing.sparse import sparse_zscore_column
from exprmat.preprocessing.filter import highly_variable_genes_ff, stable_genes
from exprmat.ansi import warning, info


def normalize(adata, counts = 'X', dest = None, method = 'total', **kwargs):

    matrix = adata.X if counts == 'X' else adata.layers[counts]
    norm = normalize_linear_methods[method](matrix, **kwargs)

    if dest is not None: adata.layers[dest] = norm
    elif counts == 'X': adata.X = norm
    else: adata.layers[counts] = norm
    return adata


def highly_variable(adata, counts = 'X', dest = 'hvg', method = 'ff', **kwargs):

    m_count = adata.X if counts == 'X' else adata.layers[counts]
    if method == 'ff':
        indices = highly_variable_genes_ff(m_count, **kwargs)
        adata.var[dest] = index_to_bool(indices, n = adata.n_vars)
    else: warning(f'invalid hvg selection method: {method}')
    return adata


def log_transform(adata, norm = 'norm', dest = 'lognorm'):

    m_norm = adata.X if norm == 'X' else adata.layers[norm]
    adata.layers[dest] = log1p(m_norm)
    return adata


def scale(adata, lognorm = 'lognorm', dest = 'scaled', gene_mean = None, gene_stdev = None):
    
    m_lognorm = adata.X if lognorm == 'X' else adata.layers[lognorm]
    adata.layers[dest] = np.array(sparse_zscore_column(
        m_lognorm, 
        gene_mean = gene_mean, 
        gene_stdev = gene_stdev
    ))
    return adata


def index_to_bool(indices, n):
    falses = np.zeros(n).astype(bool)
    falses[indices] = True
    return falses


def bool_to_index(bools):
    return np.nonzero(bools)[0]