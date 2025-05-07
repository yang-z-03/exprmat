
import numpy as np

from exprmat.preprocessing.linear import normalize_linear_methods
from exprmat.preprocessing.normal import log1p
from exprmat.preprocessing.uniform import scale as scale_u
from exprmat.preprocessing.uniform import sct
from exprmat.preprocessing.filter import (
    highly_variable_genes_ff,
    highly_variable_genes_seurat,
    highly_variable_nb,
    highly_variable_nb_log
)
from exprmat.ansi import warning, info
from exprmat.utils import choose_layer


def normalize(adata, counts = 'X', dest = None, method = 'total', **kwargs):

    matrix = adata.X if counts == 'X' else adata.layers[counts]
    norm = normalize_linear_methods[method](matrix, **kwargs)

    if dest is not None: adata.layers[dest] = norm
    elif counts == 'X': adata.X = norm
    else: adata.layers[counts] = norm
    return adata


def log_transform(adata, norm = 'norm', dest = 'lognorm'):

    m_norm = adata.X if norm == 'X' else adata.layers[norm]
    adata.layers[dest] = log1p(m_norm)
    return adata


def scale(adata, lognorm = 'lognorm', dest = 'scaled', gene_mean = None, gene_stdev = None):
    
    m_lognorm = adata.X if lognorm == 'X' else adata.layers[lognorm]
    adata.layers[dest] = np.array(scale_u(
        m_lognorm, 
        gene_mean = gene_mean, 
        gene_stdev = gene_stdev
    ))
    return adata


def sctransform(
    adata, counts = 'X', latent = ['log.umi'],
    dest_scaled = 'scaled', 
    min_cells = 5, min_variance = float('-inf'), bw_adjust = 3, clip_range = None,
    dest_norm = 'norm', compute_norm = True,
    dest_lognorm = 'lognorm', compute_lognorm = False
):
    
    m_c = adata.X if counts == 'X' else adata.layers[counts]
    residuals, params, _ = sct(
        m_c, adata.var, latent_variable = latent,
        min_cells = min_cells, min_variance = min_variance, 
        bw_adjust = bw_adjust, clip_range = clip_range
    )

    adata.layers[dest_scaled] = residuals
    if compute_norm: adata.layers[dest_norm] = residuals
    if compute_lognorm: log_transform(adata, norm = dest_norm, dest = dest_lognorm)
    return adata


def highly_variable(
    adata, counts = 'X', lognorm = 'lognorm', dest = 'vst', method = 'vst',
    n_top_genes: int | None = 2000, min_disp: float = 0.5, max_disp: float = np.inf,
    min_mean: float = 0.0125, max_mean: float = 3, use_bounds = True,
    **kwargs
):

    m_count = adata.X if counts == 'X' else adata.layers[counts]

    # signal to noise ratio (fano factor) on binned means
    if method == 'ff':
        indices = highly_variable_genes_ff(m_count, **kwargs)
        adata.var[dest + '.hvg'] = index_to_bool(indices, n = adata.n_vars)
    
    # method based on sctransform style vst.
    # implemented in >= v3 version of seurat.
    elif method == 'vst':
        df = highly_variable_genes_seurat(
            m_count, obs = adata.obs, var = adata.var, subset = False,
            **kwargs
        )
        adata.var[dest + '.means'] = df['means']
        adata.var[dest + '.vars'] = df['variances']
        adata.var[dest + '.vars.norm'] = df['var.norm']
        adata.var[dest + '.hvg.rank'] = df['hvg.rank']
        adata.var[dest + '.hvg'] = df['hvg']

    # method implemented in cellranger's selection of hvg.
    elif method == 'cellranger':
        m_lognorm = choose_layer(adata, lognorm)
        df = highly_variable_nb_log(m_lognorm, adata.var, **kwargs)

        # here, we get the full ranked list for all genes in the dataset.
        # we will here manually apply selection criteria to construct hvgs.
        adata.var[dest + '.means'] = df['means']
        adata.var[dest + '.dispersions'] = df['dispersions']
        adata.var[dest + '.dispersions.norm'] = df['norm.disp']
        if use_bounds:
            adata.var[dest + '.hvg'] = (
                df['means'] > min_mean & 
                df['means'] < max_mean & 
                df['norm.disp'] > min_disp &
                df['norm.disp'] < max_disp
            )

        else:
            if n_top_genes > adata.n_vars:
                info(f'you attempt to get {n_top_genes} hvgs from a dataset with only {adata.n_vars} genes.')
                info(f'returning all genes as highly variable. you may run again with bounding criteria.')
                n_top_genes = adata.n_vars
            
            disp_cut_off = nth_highest(df['norm.disp'].to_numpy(), n_top_genes)
            adata.var[dest + '.hvg'] = np.nan_to_num(df['norm.disp'].to_numpy(), nan=-np.inf) >= disp_cut_off

    # the method implemented in v1 and v2 seurat.
    # very similar to the ff method, which also takes binned means
    elif method == 'binned':
        df = highly_variable_nb(m_count, adata.var, **kwargs)

        adata.var[dest + '.means'] = df['means']
        adata.var[dest + '.dispersions'] = df['dispersions']
        adata.var[dest + '.dispersions.norm'] = df['norm.disp']
        if use_bounds:
            adata.var[dest + '.hvg'] = (
                df['means'] > min_mean & 
                df['means'] < max_mean & 
                df['norm.disp'] > min_disp &
                df['norm.disp'] < max_disp
            )

        else:
            if n_top_genes > adata.n_vars:
                info(f'you attempt to get {n_top_genes} hvgs from a dataset with only {adata.n_vars} genes.')
                info(f'returning all genes as highly variable. you may run again with bounding criteria.')
                n_top_genes = adata.n_vars
            
            disp_cut_off = nth_highest(df['norm.disp'].to_numpy(), n_top_genes)
            adata.var[dest + '.hvg'] = np.nan_to_num(df['norm.disp'].to_numpy(), nan=-np.inf) >= disp_cut_off

    else: warning(f'invalid hvg selection method: {method}')
    return adata


def index_to_bool(indices, n):
    falses = np.zeros(n).astype(bool)
    falses[indices] = True
    return falses


def bool_to_index(bools):
    return np.nonzero(bools)[0]


def nth_highest(x, n: int) -> float:
    x = x[~ np.isnan(x)]
    if n > x.size: n = x.size
    x[::-1].sort()
    return x[n - 1]