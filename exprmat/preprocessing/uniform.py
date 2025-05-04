'''
Normalization in uniform normal space.
'''

import numpy as np
from exprmat.preprocessing.sparse import sparse_zscore_column
from exprmat.preprocessing.sctransform.transform import vst


def scale(E, gene_mean = None, gene_stdev = None):
    return sparse_zscore_column(E, gene_mean, gene_stdev)


def sct(
    E, var = None, latent_variable = ['log.umi'], 
    min_cells = 5, min_variance = float('-inf'), bw_adjust = 3,
    clip_range = None
):
    return vst(
        E, cell_attr = var, latent_var = latent_variable, method = 'poisson',
        residual_type = 'pearson', min_cells = min_cells, min_variance = min_variance,
        bw_adjust = bw_adjust, res_clip_range = clip_range
    )