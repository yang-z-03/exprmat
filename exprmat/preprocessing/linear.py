'''
Normalization in linear space
'''

import scipy
import scipy.sparse as sp
import numpy as np
import scanpy as sc

from exprmat.ansi import warning, info


def normalize_total(
    E, total_counts = None, 
    exclude_dominant_frac = 1, included_genes = [], 
    target_total = None
):
    
    ''' 
    Cell-level total counts normalization of input counts matrix, excluding overly 
    abundant genes if desired. Return normalized counts, average total counts, and 
    (if exclude_dominant_frac < 1) list of genes used to calculate total counts 

    Parameters
    -----------
    
    E : np.array | scipy.sparse.csr_matrix | scipy.sparse.csc_matrix
        Array of raw expression matrix in counts. This is either sparse or dense.

    total_counts : None | int
        Per-cell total expression counts. If omitted, will calculate as row sums.
    
    exclude_dominant_frac : float
        Exclude the gene if its expression fraction makes up more than this value
        in any of the cells. (For extensively dominant genes may distort the overall
        normalization process)
    
    included_genes : List
        Manually specify included genes. If left to empty, we will calculate dominant
        genes from ``exclude_dominant_frac`` as described above.
    
    target_total : int
        The normalizing total counts. One may think 1e6 as acceptable for 10x
        based scRNA-seq experiments, but we left it None as default for making the
        mean counts per cell as the target total sum.
    '''

    E = E.tocsc() # convert to csc matrix for faster speed of gene indexing.
    ncell = E.shape[0]
    if total_counts is None:
        if len(included_genes) == 0:
            if exclude_dominant_frac == 1:
                # total number of umi's in a cell
                tots_use = E.sum(axis = 1)
            else:
                tots = E.sum(axis = 1)
                wtmp = scipy.sparse.lil_matrix((ncell, ncell))
                wtmp.setdiag(1. / tots)
                # genes that present to be dominant in any of the cells
                dominant_genes = ((wtmp * E) > exclude_dominant_frac).sum(axis = 0) > 0
                included_genes = np.asarray(~dominant_genes)[0,:]
                tots_use = E[:, included_genes].sum(axis = 1)
                info(f'{np.sum(~ included_genes)} genes seems to be dominant '
                     f'(> {100 * exclude_dominant_frac:.1f}%) in at least 1 of the cells.')
        else: tots_use = E[:, included_genes].sum(axis = 1)
    else: tots_use = total_counts.copy()

    if target_total is None:
        target_total = np.mean(tots_use)

    b = np.array(float(target_total) / tots_use.T)
    e_norm = mul_row(E, b)

    return e_norm


def mul_row(E, b):

    if len(b.shape) == 2: b = b[0]
    if not sp.issparse(E):
        # this will densify the matrix, causing incredibly large memory comsumption
        # when treating with an sparse matrix.
        E = sp.csr_matrix(E)
        
    if sp.isspmatrix_csr(E):
        e_norm = E.copy()
        e_norm.data *= b.repeat(np.diff(E.indptr))
    elif sp.isspmatrix_coo(E):
        e_norm = E.copy()
        e_norm.data *= b[E.row]
    elif sp.isspmatrix_csc(E):
        e_norm = E.copy()
        e_norm.data *= b[E.indices]
    
    return e_norm


def normalize_size_factor(x, size_factor = None):
    result = x / (x.sum() / 1000000.0)
    if size_factor is not None: result /= size_factor
    return result


def get_size_factor_for_regions(regions):
    def size(x):
        x = x.split(':')[1].split("-")
        return int(x[1]) - int(x[0])
    return np.array(list(size(x) for x in regions))


def rpm(E):
    return normalize_size_factor(E)


normalize_linear_methods = {
    'total': normalize_total,
    'rpm': rpm
}