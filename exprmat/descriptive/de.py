'''
DE (Differential expression models currently use method wrapped by scanpy package.)
Scanpy does not support covariate regression in the differential testing process
though, I will add these routines to test statisticals with more modelling methods
beyond scanpy here. But not now.
'''

from exprmat.plotting.de import get_marker_df


def markers(
    adata, groupby, *,
    mask_var = None, groups = 'all', reference = 'rest', 
    n_genes = None, rankby_abs = False, 
    pts = True, key_added = 'markers', method = 't-test',
    corr_method = 'benjamini-hochberg', tie_correct = False, 
    layer = 'X', gene_symbol = 'gene', **kwds
):
    
    import scanpy as sc
    if layer == 'X': layer = None

    sc.tl.rank_genes_groups(
        adata = adata, groupby = groupby,
        mask_var = mask_var, groups = groups, reference = reference,
        n_genes = n_genes, rankby_abs = rankby_abs, pts = pts,
        key_added = key_added, copy = False, method = method, corr_method = corr_method,
        tie_correct = tie_correct, layer = layer, **kwds
    )

    groups = adata.obs[groupby].value_counts().index.tolist() if groups == 'all' else groups
    result_table = {}
    result_table['params'] = adata.uns[key_added]['params']
    result_table['names'] = adata.uns[key_added]['names']
    result_table['differential'] = {}

    for g in groups:
        result_table['differential'][g] = \
            get_marker_df(adata, key = key_added, group = g, gene_symbols = gene_symbol)

    del adata.uns[key_added]
    adata.uns[key_added] = result_table
    pass