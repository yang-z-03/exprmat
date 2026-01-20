
from __future__ import annotations
from functools import reduce

import anndata
import pandas
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import norm
from mudata import MuData

from exprmat.utils import mdata_to_adata_2
from exprmat.ansi import error, warning, info
from exprmat.lr.preprocess import prep_check_adata, assert_covered, filter_resource
from exprmat.lr.utils import join_stats, get_props, get_groupby_subset
from exprmat.lr.resources import handle_resource
from exprmat.lr.resources import explode_complexes, filter_reassemble_complexes
from exprmat.lr.utils import get_means_permutations, get_mat_idx
from exprmat.lr.aggregate import aggregate
from exprmat.lr.utils import (
    common_method_columns as M, 
    default_common_columns as C, 
    default_primary_columns as P, 
    internal_values as I
)


def pipeline(
    adata: anndata.AnnData,
    groupby: str,
    taxa_source : str,
    taxa_dest : str,
    resource_name: str,
    resource: pd.DataFrame | None,
    interactions,
    groupby_pairs: pd.DataFrame | None,
    expr_prop: float,
    min_cells: int,
    base: float,
    de_method: str,
    n_perms: int,
    seed: int,
    verbose: bool,
    use_raw: bool,
    n_jobs: int,
    layer: str | None,
    supp_columns: list | None = None,
    return_all_lrs: bool = False,
    method_meta = None,
    consensus_methods: list = None,
    consensus_opts: list = None,
    aggregate_method: str | None = None,
    mdata_kwargs: dict = dict(),
):
    """
    Parameters
    ----------

    adata : anndata.AnnData
        Annotated data object.

    groupby : str
        The key of the observations grouping to consider.

    resource_name : str
        Name of the resource to be loaded and use for ligand-receptor inference.
        Also specify `taxa_source` to find resource within certain taxa.

    resource : DataFrame
        Parameter to enable external resources to be passed. Expects a pandas dataframe
        with [`ligand`, `receptor`] columns. None by default. If provided will override
        the resource requested via `resource_name`

    expr_prop : float
        Minimum expression proportion for the ligands/receptors (and their subunits) in the
        corresponding cell identities. Set to `0`, to return unfiltered results.

    min_cells : int
        Minimum cells per cell identity

    base : float
        The base by which to do expm1 (relevant only for 1 vs rest log fold change calculation)
    
    de_method : str
        Differential expression method. `scanpy.tl.rank_genes_groups` is used to rank genes
        according to 1-vs-rest. The default method is 't-test'. Only relevant if p-values
        are included in `supp_cols`

    n_perms : int
        n permutations (relevant only for permutation-based methods)

    seed : int
        Random seed for reproducibility

    layer : str
        Layer in to use. If None, use `X`.

    supp_columns : list
        additional columns to be added to the output of each method.

    return_all_lrs : bool
        Whether to return all LRs, or only those that surpass the expr_prop threshold.

    method_meta : lr_method
        Instance of `lr_method` classes, by default is None, and returns the raw statistics
        table rather than the filtered ligand-receptor scores.

    consensus_methods : list
        Methods to consider when running in consensus mode.

    consensus_opts : list[str]
        Ways to aggregate interactions across methods by default does all aggregations 
        (i.e. ['Specificity', 'Magnitude']).

    aggregate_method : Literal['rra', 'mean]
        Robust rank aggregation ('rra') or mean rank ('mean').
    """
    _key_cols = P.primary

    if method_meta is not None:
        _complex_cols, _add_cols = method_meta.complex_cols, method_meta.add_cols
    else:
        _complex_cols = [C.ligand_means, C.receptor_means]
        _add_cols = M.get_all_values()

    if n_perms is None:
        consensus_opts = 'Magnitude'

    if supp_columns is None:
        supp_columns = []
    _add_cols = _add_cols + [P.ligand, P.receptor,
                             C.ligand_props, C.receptor_props] + supp_columns

    # initialize mat_mean for sca
    mat_mean = None
    mat_max = None

    resource = handle_resource(
        source_taxa = taxa_source, destination_taxa = taxa_dest,
        interactions = interactions,
        resource = resource,
        resource_name = resource_name,
        verbose = verbose
    )

    resource = explode_complexes(resource)

    if isinstance(adata, MuData):
        adata = mdata_to_adata_2(adata, **mdata_kwargs, verbose=verbose)
        use_raw = False
        layer = None

    groupby_subset = get_groupby_subset(groupby_pairs = groupby_pairs)
    adata = prep_check_adata(
        adata = adata,
        groupby = groupby,
        groupby_subset = groupby_subset,
        min_cells = min_cells,
        use_raw = use_raw,
        layer = layer
    )

    if M.mat_mean in _add_cols:
        mat_mean = np.mean(adata.X, dtype = 'float32')

    # get mat max for cellchat
    if M.mat_max in _add_cols:
        mat_max = adata.X.max()
        assert isinstance(mat_max, np.float32)

    # check overlap between resource and adata
    assert_covered(
        np.union1d(np.unique(resource[P.ligand]),
        np.unique(resource[P.receptor])),
        adata.var_names, verbose = verbose
    )

    # filter Resource
    resource = filter_resource(resource, adata.var_names)

    # cluster stats
    if (M.ligand_cdf in _add_cols) or (M.receptor_cdf in _add_cols):
        clstats = cluster_stats(adata)

    # create entities
    entities = np.union1d(
        np.unique(resource[P.ligand]),
        np.unique(resource[P.receptor])
    )

    # filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)].copy()

    if verbose: info(f"generating ligand-receptor stats for {adata.shape[0]} * {adata.shape[1]}")

    # Get lr results
    lr_res = get_lr(
        adata = adata,
        resource = resource,
        groupby_pairs = groupby_pairs,
        mat_mean = mat_mean,
        mat_max = mat_max,
        relevant_cols = _key_cols + _add_cols + _complex_cols,
        de_method = de_method,
        base = base,
        verbose = verbose
    )

    # ligand and receptor score based on unfiltered cluster mean and cluster 
    # std. handles protein complexes
    if (M.ligand_cdf in _add_cols) or (M.receptor_cdf in _add_cols):
        lr_res = complex_score(lr_res, clstats)

    # mean sums required for NATMI (note done on subunits also)
    if M.ligand_means_sums in _add_cols:
        on = [x for x in P.complete if x != P.source]
        lr_res = sum_means(lr_res, what = C.ligand_means, on = on)
    if M.receptor_means_sums in _add_cols:
        on = [x for x in P.complete if x != P.target]
        lr_res = sum_means(lr_res, what = C.receptor_means, on = on)

    # calculate score
    if method_meta is not None:
        if method_meta.method_name == "ra":
            # Run all methods in consensus
            lrs = {}
            for method in consensus_methods:
                if verbose: info(f"running {method.method_name}")

                lrs[method.method_name] = \
                    run_method(
                        lr_res = lr_res.copy(),
                        adata = adata,
                        expr_prop = expr_prop,
                        _score = method,
                        _key_cols = _key_cols,
                        _complex_cols = method.complex_cols,
                        _add_cols = method.add_cols,
                        n_perms = n_perms,
                        seed = seed,
                        return_all_lrs = return_all_lrs,
                        n_jobs = n_jobs,
                        verbose = verbose,
                        _aggregate_flag = True
                    )
                
            if consensus_opts is not False:
                lr_res = aggregate(
                    lrs, consensus = method_meta, aggregate_method = aggregate_method,
                    key_cols = _key_cols, consensus_opts = consensus_opts,
                )

            else: return lrs

        else:
            lr_res = run_method(
                lr_res = lr_res, adata = adata, expr_prop = expr_prop,
                _score = method_meta, _key_cols = _key_cols,
                _complex_cols = _complex_cols,
                _add_cols = _add_cols,
                n_perms = n_perms,
                return_all_lrs = return_all_lrs,
                n_jobs = n_jobs, verbose = verbose, seed = seed
            )

    else:  # just return lr_res
        lr_res = filter_reassemble_complexes(
            lr_res = lr_res, key_columns = _key_cols, expr_prop = expr_prop,
            complex_cols = _complex_cols, return_all_lrs = return_all_lrs
        )

    if method_meta is not None:
        orderby, ascending = (method_meta.magnitude, method_meta.magnitude_ascending) \
            if method_meta.magnitude is not None \
            else (method_meta.specificity, method_meta.specificity_ascending)
        lr_res = lr_res.sort_values(by = orderby, ascending = ascending)

    return lr_res


def get_lr(
    adata, resource, groupby_pairs, relevant_cols, 
    mat_mean, mat_max, de_method, base, verbose
):
    
    labels = adata.obs[I.label].cat.categories

    # method-specific stats
    connectome_flag = (M.ligand_zscores in relevant_cols) | (
                M.receptor_zscores in relevant_cols)
    
    if connectome_flag:
        adata.layers['scaled'] = sc.pp.scale(adata, copy = True).X

    logfc_flag = (M.ligand_logfc in relevant_cols) | (
        M.receptor_logfc in relevant_cols)
    
    if logfc_flag:
        if 'log1p' in adata.uns_keys():
            if (adata.uns['log1p']['base'] is not None) & verbose:
                info("assuming that counts were log transformed with base e.")
        elif ('log1p' not in adata.uns_keys()) & verbose:
            info("assuming that counts were log transformed with base e.")
        
        adata.layers['norm'] = adata.X.copy()
        adata.layers['norm'].data = expm1_base(adata.X.data, base)

    # initialize dict
    dedict = {}

    # Calc pvals + other stats per gene or not
    rank_genes_bool = (C.ligand_pvals in relevant_cols) | (C.receptor_pvals in relevant_cols)
    if rank_genes_bool:
        adata = sc.tl.rank_genes_groups(
            adata, groupby = I.label, method = de_method, use_raw = False, copy = True)

    for label in labels:
        temp = adata[adata.obs[I.label] == label, :]
        a = get_props(temp.X)
        stats = pd.DataFrame({'names': temp.var_names, 'props': a}). \
            assign(label = label).sort_values('names')
        if rank_genes_bool:
            pvals = sc.get.rank_genes_groups_df(adata, label)
            stats = stats.merge(pvals)
        dedict[label] = stats

    # check if genes are ordered correctly
    if not list(adata.var_names) == list(dedict[labels[0]]['names']):
        error("variable names did not match de results.")

    # Calculate Mean, logFC and z-scores by group
    for label in labels:
        temp = adata[adata.obs[I.label].isin([label])]
        dedict[label]['means'] = temp.X.mean(axis=0).A.flatten()
        if connectome_flag:
            dedict[label]['zscores'] = temp.layers['scaled'].mean(axis = 0)
        if logfc_flag:
            dedict[label]['logfc'] = log2fc(adata, label)
        if isinstance(mat_max, np.float32): # cellchat flag
            dedict[label]['trimean'] = trimean(temp.X / mat_max)

    pairs = (pd.DataFrame(
        np.array(np.meshgrid(labels, labels))
            .reshape(2, np.size(labels) * np.size(labels)).T)
            .rename(columns={0: P.source, 1: P.target}))

    if groupby_pairs is not None:
        pairs = pairs.merge(groupby_pairs, on = [P.source, P.target], how = 'inner')

    # join Stats
    lr_res = pd.concat(
        [join_stats(source, target, dedict, resource) for source, target in
         zip(pairs[P.source], pairs[P.target])]
    )

    if M.mat_mean in relevant_cols:
        assert isinstance(mat_mean, np.float32)
        lr_res[M.mat_mean] = mat_mean

    # NOTE: this is not needed
    if isinstance(mat_max, np.float32):
        lr_res[M.mat_max] = mat_max

    # subset to only relevant columns
    relevant_cols = np.intersect1d(relevant_cols, lr_res.columns)

    return lr_res[relevant_cols]


def sum_means(lr_res, what, on):
    return lr_res.join(lr_res.groupby(on)[what].sum(), on = on, rsuffix = '.sums')


def log2fc(adata, label) -> np.ndarray:
    # get subject vs rest cells
    subject = adata[adata.obs[I.label].isin([label])]
    rest = adata[~adata.obs[I.label].isin([label])]

    # subject and rest means
    subj_means = subject.layers['norm'].mean(0).A.flatten()
    rest_means = rest.layers['norm'].mean(0).A.flatten()

    # log2 + 1 transform
    subj_log2means = np.log2(subj_means + 1)
    loso_log2means = np.log2(rest_means + 1)

    logfc_vec = subj_log2means - loso_log2means
    return logfc_vec


def expm1_base(X, base):
    return np.power(base, X) - 1


def run_method(
    lr_res: pandas.DataFrame,
    adata: anndata.AnnData,
    expr_prop: float,
    _score,
    _key_cols: list,
    _complex_cols: list,
    _add_cols: list,
    n_perms: int,
    seed: int,
    return_all_lrs: bool,
    n_jobs: int,
    verbose: bool,
    _aggregate_flag: bool = False  # Indicates whether we're generating the consensus
) -> pd.DataFrame:
    
    # re-assemble complexes - specific for each method
    lr_res = filter_reassemble_complexes(
        lr_res = lr_res, key_columns = _key_cols, expr_prop = expr_prop,
        return_all_lrs = return_all_lrs, complex_cols = _complex_cols
    )

    _add_cols = _add_cols + [P.ligand, P.receptor]
    relevant_cols = reduce(np.union1d, [_key_cols, _complex_cols, _add_cols])
    if return_all_lrs:
        relevant_cols = list(relevant_cols) + [I.lrs_to_keep]
        # separate those that pass from rest
        rest_res = lr_res[~lr_res[I.lrs_to_keep]]
        rest_res = rest_res[relevant_cols]
        lr_res = lr_res[lr_res[I.lrs_to_keep]]

    lr_res = lr_res[relevant_cols]

    if (M.mat_max in _add_cols) & (_score.method_name == "cellchat"):
        norm_factor = np.unique(lr_res[M.mat_max].values)[0]
        agg_fun = trimean # calculate sparse matrix quantiles?
    
    else:
        norm_factor = None
        agg_fun = np.mean # NOTE: change to sparse matrix mean?

    if _score.permute:
        # get permutations
        if n_perms is not None:
            perms = get_means_permutations(
                adata = adata, n_perms = n_perms, seed = seed,
                agg_fun = agg_fun, norm_factor = norm_factor,
                n_jobs = n_jobs, verbose = verbose
            )

            # get tensor indexes for ligand, receptor, source, target
            ligand_idx, receptor_idx, source_idx, target_idx = get_mat_idx(adata, lr_res)

            # ligand and receptor perms
            ligand_stat_perms = perms[:, source_idx, ligand_idx]
            receptor_stat_perms = perms[:, target_idx, receptor_idx]
            # stack them together
            perm_stats = np.stack((ligand_stat_perms, receptor_stat_perms), axis=0)
            
        else:
            perm_stats = None
            _score.specificity = None

        scores = _score.fun(x = lr_res, perm_stats = perm_stats)
    else: scores = _score.fun(x = lr_res)

    lr_res.loc[:, _score.magnitude] = scores[0]
    lr_res.loc[:, _score.specificity] = scores[1]

    if return_all_lrs:
        # re-append rest of results
        lr_res = pd.concat([lr_res, rest_res], copy=False)
        if _score.magnitude is not None:
            fill_value = assign_min_max(lr_res[_score.magnitude], _score.magnitude_ascending)
            lr_res.loc[~lr_res[I.lrs_to_keep], _score.magnitude] = fill_value
        if _score.specificity is not None:
            fill_value = assign_min_max(lr_res[_score.specificity], _score.specificity_ascending)
            lr_res.loc[~lr_res[I.lrs_to_keep], _score.specificity] = fill_value

    if _aggregate_flag: # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res


def assign_min_max(x, x_ascending):
    if x_ascending: return np.max(x)
    else: return np.min(x)


def trimean(a, axis = 0):
    quantiles = np.quantile(a.toarray(), q=[0.25, 0.75], axis = axis)
    median = np.median(a.toarray(), axis = axis)
    return (quantiles[0] + 2 * median + quantiles[1]) / 4

def cluster_stats(adata):
    clstats = adata.obs.groupby(I.label, observed = False).size().to_frame(name = 'counts')
    labels = adata.obs[I.label].cat.categories
    
    for label in labels:
        temp = adata[adata.obs[I.label].isin([label])]
        clstats.loc[label, 'mean'] = temp.X.mean()
        clstats.loc[label, 'std'] = np.std(temp.X.toarray())

    return clstats


def cdf(gene_mean, cluster_mean, cluster_std, cluster_counts):

    probability = norm.cdf(
        gene_mean, 
        loc = cluster_mean, 
        scale = cluster_std / np.sqrt(cluster_counts)
    )

    probability[gene_mean == 0] = 0
    return probability


def complex_score(lr_res, clstats):

    tmp = lr_res.merge(
        clstats.add_prefix('source.'), 
        left_on = 'source', right_index = True, how = 'left')
    tmp = tmp.merge(
        clstats.add_prefix('target.'), 
        left_on = 'target', right_index = True, how = 'left')

    lr_res[M.ligand_cdf] = cdf(
        tmp['ligand.means'],
        tmp['source.mean'],
        tmp['source.std'],
        tmp['source.counts']
    )

    lr_res[M.receptor_cdf] = cdf(
        tmp['receptor.means'],
        tmp['target.mean'],
        tmp['target.std'],
        tmp['target.counts']
    )

    return lr_res