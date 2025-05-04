'''
Quality control routines is done on sample level. A sample-level expression matrix indicates a 
cDNA library, and may be within the same batch as others. This is the smallest unit of expression
matrix integration. Variations introduced during library preparation and sample acquisition should
be reflected in these (as well as the batch effect), and is only left-out exceptional for mixed
samples from different biological origins.
'''

import os
from functools import reduce
from random import seed, sample
from typing import Union, Optional, Sequence, Tuple, List, Dict

import pandas as pd 
import anndata
import numpy as np 
import scanpy as sc
from scipy.sparse import issparse

from exprmat.reader.experiment import experiment
from exprmat.ansi import warning, info
from exprmat.preprocessing.doublets.scrublet import scrublet_init, scrublet


# here are several methods for outlier detection over a one-dimensional distribution:

def mads(x, nmads = 5):
    '''
    Threshold on median absolute deviation (MAD) of array ``x``.
    '''
    mad = np.median(np.absolute(x - np.median(x)))
    t1 = np.median(x) - (nmads * mad)
    t2 = np.median(x) + (nmads * mad)
    return t1, t2


def tukey(x, n = 1.5):
    '''
    Threshold on inter-quartile range of array ``x``.
    '''
    lower = np.quantile(x, 0.25)
    upper = np.quantile(x, 0.75)
    iqr = upper - lower
    return (lower - n * iqr, upper + n * iqr)


def is_outliers(x, method = 'mads', n = 5):
    '''
    Given a certain array, it returns a boolean array with True values only at indeces 
    from entries within threshold ranges (for specified method ``tukey`` or ``mads``).
    '''
    if method == 'mads': tresholds = mads(x, nmads = n)
    elif method == 'tukey': thresholds = tukey(x, n = n)
    return (x > tresholds[0]) & (x < tresholds[1])


def rna_qc(
    adata: anndata.AnnData,
    sample: str = '?',

    # mitochondrial quality filter
    mt_seqid = 'MT',
    mt_percent = 0.15,
    ribo_genes = None,
    ribo_percent = None,

    # outlier quality filter
    outlier_mode = 'tukey',
    outlier_n = 1.5,
    
    # doublet filter
    doublet_method = 'scrublet',

    # low depth quality filter
    min_cells = 3,
    min_genes = 300
):
    '''
    Quality control filtering on experiment object.

    This is a routine pipeline for common tasks for filtering out inadequately sequenced
    cells. The routine consists of the following steps:

    *   Remove cells that have exceptionally high mitochondrial gene fraction, which means
        that the cells have few nuclear RNA detection and may be of peculiar states.
    *   Calculate ribosomal gene fraction, but perform no exclusion based on this.
    *   Detect outliers that have exceptionally high or low reads (UMIs) and gene detection.
    *   Detect doublets and multiplets, and remove them.
    *   Remove genes and cells that are inadequately sequenced.

    This quality control should be sample-wise. So that this routine is carried out on
    the `modalities` field of the experiment object. You may want to re-integrate the 
    dataset after quality control, or redo it if an old one exists.

    Mitochondrial genes can be distinguished by a unique sequence in the genome annotation,
    however, ribosomal genes are scattered around the genome, and can have no uniform
    nomenclature (commonly though, be named after Rpl and Rps) You'd better specify
    the ``ribo_genes`` array to proper ENSEMBL IDs or else the program will guess from the names.
    '''
    
    # manual calculations of ribosomal and mitochondrial genes
    mask_mito = [x == mt_seqid for x in adata.var['.seqid'].tolist()]
    mask_ribo = [False if isinstance(x, float) else \
                    x.lower().startswith('rps') or x.lower().startswith('rpl') \
                    for x in adata.var['gene'].tolist()] if ribo_genes is None else \
                [x in ribo_genes for x in adata.var['gene'].tolist()]

    import numpy as np
    info(f'found {np.sum(np.array(mask_mito))} mitochondrial genes (expected 13)')
    info(f'found {np.sum(np.array(mask_ribo))} ribosomal genes')

    print(f'quality controlling sample [{sample}] ...')
    print(f'raw dataset contains {adata.n_obs} cells, {adata.n_vars} genes')

    # quality control rna seq.
    adata.var['n.umi'] = np.sum(adata.X, axis = 0).tolist()[0]
    adata.obs['n.umi'] = np.sum(adata.X, axis = 1).transpose().tolist()[0]
    adata.var['n.cells'] = np.sum(adata.X > 0, axis = 0).tolist()[0]
    adata.obs['n.genes'] = np.sum(adata.X > 0, axis = 1).transpose().tolist()[0]
    adata.obs['n.mito'] = np.sum(adata.X[:, mask_mito], axis = 1).transpose().tolist()[0]
    adata.obs['n.ribo'] = np.sum(adata.X[:, mask_ribo], axis = 1).transpose().tolist()[0]
    adata.obs['pct.mito'] = adata.obs['n.mito'] / adata.obs['n.umi']
    adata.obs['pct.ribo'] = adata.obs['n.ribo'] / adata.obs['n.umi']

    # detect outliers
    if outlier_mode == 'mads':
        umi_lower, umi_upper = mads(adata.obs['n.umi'].to_numpy(), nmads = outlier_n)
    elif outlier_mode == 'tukey':
        umi_lower, umi_upper = tukey(adata.obs['n.umi'].to_numpy(), n = outlier_n)
    else: umi_lower, umi_upper = 200, 100000

    f_obs = \
        (adata.obs['n.umi'] <= umi_upper) & \
        (adata.obs['pct.mito'] < mt_percent) & \
        (adata.obs['n.umi'] >= umi_lower) & \
        (adata.obs['n.genes'] >= min_genes)
    
    if ribo_percent is not None:
        f_obs = f_obs & (adata.obs['pct.ribo'] < ribo_percent)

    f_var = adata.var['n.cells'] >= min_cells

    # doublet detection using the filtered expression matrix.
    subset = adata[f_obs, f_var].copy()
    subset.obs['qc'] = True
    
    if doublet_method == 'scrublet':
        scrublet_init(subset, random_state = 42)
        scrublet(subset)
    
    adata.obs['filter'] = f_obs
    adata.obs['score.doublet'] = 0.0
    adata.obs['score.doublet.se'] = 0.0
    adata.obs['is.doublet'] = False

    adata.obs.loc[subset.obs['score.doublet'].index, 'score.doublet'] = \
        subset.obs['score.doublet']
    adata.obs.loc[subset.obs['score.doublet.se'].index, 'score.doublet.se'] = \
        subset.obs['score.doublet.se']
    if 'is.doublet' in subset.obs.columns.tolist():
        adata.obs.loc[subset.obs['score.doublet'].index, 'is.doublet'] = \
            subset.obs['is.doublet']
    
    adata.uns['scrublet'] = subset.uns['scrublet']
    adata.obs['qc'] = adata.obs['filter']
    adata.var['qc'] = f_var

    if 'is.doublet' in subset.obs.columns.tolist(): 
        adata.obs['qc'] = adata.obs['qc'] & (~adata.obs['is.doublet'])
    print(f'filtered dataset contains {(adata.obs["qc"].to_numpy().sum())} cells, ' +
          f'{adata.var["qc"].to_numpy().sum()} genes')
    del subset
        
    return adata


def filter_cells(
    adata: anndata.AnnData,
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None
):
    '''
    Filter cell outliers based on counts and numbers of genes expressed. For instance, only keep 
    cells with at least ``min_counts`` counts or ``min_genes`` genes expressed. This is to filter 
    measurement outliers, i.e. “unreliable” observations. You should only provide one of the 
    optional parameters ``min_counts``, ``min_genes``, ``max_counts``, ``max_genes`` per call.
    '''

    mask, num = sc.pp.filter_cells(
        adata, min_genes = min_genes, min_counts = min_counts, 
        max_counts = max_counts, max_genes = max_genes, inplace = False
    )

    if min_counts is not None and max_counts is not None:
        adata.obs['n.counts'] = num
    if min_genes is not None and max_genes is not None:
        adata.obs['n.genes'] = num
    
    adata.obs['mask'] = mask
    

def filter_genes(
    adata: anndata.AnnData,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None
):
    """
    Filter genes based on number of cells or counts. Keep genes that have at least `min_counts` 
    counts or are expressed in at least `min_cells` cells or have at most `max_counts` counts 
    or are expressed in at most `max_cells` cells. Only provide one of the optional parameters 
    `min_counts`, `min_cells`, `max_counts`, `max_cells` per call.
    """

    mask, num = sc.pp.filter_genes(
        adata, min_counts = min_counts, min_cells = min_cells, 
        max_counts = max_counts, max_cells = max_cells, inplace = False)
    
    if min_counts is not None and max_counts is not None:
        adata.var['n.counts'] = num
    if min_cells is not None and min_cells is not None:
        adata.var['n.cells'] = num
    
    adata.var['mask'] = mask