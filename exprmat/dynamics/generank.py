
import numpy as np
from scipy.sparse import issparse
from exprmat.ansi import error, warning, info

from exprmat.dynamics.utils import strings_to_categoricals, vcorrcoef
from exprmat.dynamics.pseudotime import velocity_pseudotime


def get_mean_var(X, ignore_zeros = False, perc = None):

    data = X.data if issparse(X) else X
    mask_nans = np.isnan(data) | np.isinf(data) | np.isneginf(data)

    if issparse(X):
        n_nonzeros = X.getnnz(axis=0)
    else:
        n_nonzeros = (X != 0).sum(axis=0)

    if ignore_zeros:
        n_counts = n_nonzeros
    else:
        n_counts = X.shape[0]

    if mask_nans.sum() > 0:
        if issparse(X):
            data[mask_nans] = 0
            n_nans = (n_nonzeros - (X != 0).sum(0)).A1
        else:
            X[mask_nans] = 0
            n_nans = mask_nans.sum(0)
        n_counts -= n_nans

    if perc is not None:
        if np.size(perc) < 2:
            perc = [perc, 100] if perc < 50 else [0, perc]
        lb, ub = np.percentile(data, perc)
        if issparse(X):
            X.data = np.clip(data, lb, ub)
        else:
            X = np.clip(data, lb, ub)

    if issparse(X):
        mean = (X.sum(0) / n_counts).A1
        mean_sq = (X.multiply(X).sum(0) / n_counts).A1
    else:
        mean = X.sum(0) / n_counts
        mean_sq = np.multiply(X, X).sum(0) / n_counts

    n_counts = np.clip(n_counts, 2, None)  # to avoid division by zero
    var = (mean_sq - mean**2) * (n_counts / (n_counts - 1))

    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)

    return mean, var


def select_groups(adata, groups = "all", key = "leiden"):

    strings_to_categoricals(adata)
    if isinstance(groups, list) and isinstance(groups[0], int):
        groups = [f"{n}" for n in groups]
    categories = adata.obs[key].cat.categories
    groups_masks = np.array([
        categories[i] == adata.obs[key].values 
        for i, name in enumerate(categories)
    ])

    if groups == "all":
        groups = categories.values
    else:
        groups_ids = [categories.get_loc(name) for name in groups]
        groups_masks = groups_masks[groups_ids]
        groups = categories[groups_ids].values

    return groups, groups_masks


def velocity_clusters(
    adata,
    vkey = "velocity",
    match_with = "clusters",
    neighbor_key = 'neighbors',
    highly_variable = 'hvg.vst',
    disp_norm = 'disp.norm',
    likelihood = 'fit.likelihood',
    sort_by = "velocity.pseudotime",
    resolution = None,
    min_likelihood = None
):
    """
    Computes velocity clusters via louvain on velocities.

    Parameters
    ----------
    data : anndata.AnnData
        Annotated data matrix.

    vkey: str (default: `'velocity'`)
        Key of velocities computed in `tl.velocity`

    match_with : int, optional (default: 100)
        The number of genes that appear in the returned tables.

    match_with: str (default: `'clusters'`)
        Match the names of the velocity clusters with the names of this key (.obs).

    sort_by: str or None (default: `'dpt_pseudotime'`)
        Sort velocity clusters by this key (.obs).

    resolution: float (default: 0.7)
        Resolution for louvain modularity.

    min_likelihood: float between `0` and `1` or `None` (default: `None`)
        Only rank velocity of genes with a likelihood higher than min_likelihood.

    Returns
    -------
    velocity_clusters : `.obs`
        Clusters obtained from applying louvain modularity on velocity expression.
    """

    tmp_filter = ~ np.isnan(adata.layers[vkey].sum(0))
    if f"{vkey}.genes" in adata.var.keys():
        tmp_filter &= np.array(adata.var[f"{vkey}.genes"].values, dtype = bool)

    if "unspliced" in adata.layers.keys():
        n_counts = (adata.layers["unspliced"] > 0).sum(0)
        n_counts = n_counts.A1 if issparse(adata.layers["unspliced"]) else n_counts
        min_counts = min(50, np.percentile(n_counts, 50))
        tmp_filter &= np.ravel(n_counts > min_counts)

    if f"{vkey}.r2" in adata.var.keys():
        r2 = adata.var[f"{vkey}.r2"]
        min_r2 = np.percentile(r2[r2 > 0], 50)
        tmp_filter &= r2 > min_r2

    if disp_norm in adata.var.keys():
        dispersions = adata.var[disp_norm]
        min_dispersion = np.percentile(dispersions, 20)
        tmp_filter &= dispersions > min_dispersion

    if likelihood in adata.var.keys() and min_likelihood is not None:
        tmp_filter &= adata.var[likelihood] > min_likelihood

    from anndata import AnnData
    vdata = AnnData(adata.layers[vkey][:, tmp_filter])
    vdata.obs = adata.obs.copy()
    vdata.var = adata.var[tmp_filter].copy()

    if highly_variable in vdata.var.keys():
        vdata.var[highly_variable] = np.array(
            vdata.var[highly_variable], dtype = bool)

    import scanpy as sc

    sc.pp.pca(vdata, n_comps = 20, svd_solver = "arpack")
    sc.pp.neighbors(vdata, n_pcs = 20)
    sc.tl.leiden(vdata, resolution = 0.7 if resolution is None else resolution)

    if sort_by == "velocity.pseudotime" and sort_by not in adata.obs.keys():
        velocity_pseudotime(adata, vkey = vkey, neighbor_key = neighbor_key)

    if sort_by in vdata.obs.keys():
        vc = vdata.obs["leiden"]
        vc_cats = vc.cat.categories
        mean_times = [np.mean(vdata.obs[sort_by][vc == cat]) for cat in vc_cats]
        vdata.obs["leiden"] = vdata.obs["leiden"].cat.reorder_categories(
            vc_cats[np.argsort(mean_times)])

    if isinstance(match_with, str) and match_with in adata.obs.keys():
        from .utils import most_common_in_list
        vc = vdata.obs["leiden"]
        cats_nums = {cat: 0 for cat in adata.obs[match_with].cat.categories}
        for cat in vc.cat.categories:
            cells_in_cat = np.where(vc == cat)[0]
            new_cat = most_common_in_list(adata.obs[match_with][cells_in_cat])
            cats_nums[new_cat] += 1
            vc = vc.cat.rename_categories({cat: f"{new_cat} ({cats_nums[new_cat]})"})
        vdata.obs["leiden"] = vc

    else: vdata.obs["leiden"].cat.categories = np.arange(
        len(vdata.obs["leiden"].cat.categories))
    adata.obs[f"{vkey}.clusters"] = vdata.obs["leiden"].copy()

    del vdata
    return None


def rank_velocity_genes(
    data,
    vkey = "velocity",
    neighbor_key = 'neighbors',
    highly_variable = 'hvg.vst',
    disp_norm = 'disp.norm',
    likelihood = 'fit.likelihood',
    n_genes = 100,
    groupby = 'velocity.clusters',
    match_with = None,
    resolution = None,
    min_counts = None,
    min_r2 = None,
    min_corr = None,
    min_dispersion = None,
    min_likelihood = None,
    copy = False,
):
    """
    Rank genes for velocity characterizing groups.

    This applies a differential expression test (Welch t-test with overestimated
    variance to be conservative) on velocity expression, to find genes in a cluster that
    show dynamics that is transcriptionally regulated differently compared to all other
    clusters (e.g. induction in that cluster and homeostasis in remaining population).
    If no clusters are given, it priorly computes velocity clusters by applying louvain
    modularity on velocity expression.

    Parameters
    ----------
    data : anndata.AnnData
        Annotated data matrix.

    vkey: str (default: `'velocity'`)
        Key of velocities computed in `velocity()`

    n_genes : int, optional (default: 100)
        The number of genes that appear in the returned tables.

    groupby: str, list or np.ndarray (default: `None`)
        Key of observations grouping to consider.

    match_with: str or None (default: `None`)
        adata.obs key to separatively rank velocities on.

    resolution: str or None (default: `None`)
        Resolution for louvain modularity.

    min_counts: float (default: None)
        Minimum count of genes for consideration.

    min_r2: float (default: None)
        Minimum r2 value of genes for consideration.

    min_corr: float (default: None)
        Minimum Spearmans correlation coefficient between spliced and unspliced.

    min_dispersion: float (default: None)
        Minimum dispersion norm value of genes for consideration.

    min_likelihood: float between `0` and `1` or `None` (default: `None`)
        Only rank velocity of genes with a likelihood higher than min_likelihood.

    copy: bool (default: `False`)
        Return a copy instead of writing to data.
    """

    adata = data.copy() if copy else data

    if groupby is None or groupby == "velocity.clusters":
        groupby = f"{vkey}.clusters"
        velocity_clusters(
            adata,
            vkey = vkey,
            neighbor_key = neighbor_key,
            highly_variable = highly_variable,
            disp_norm = disp_norm,
            likelihood = likelihood,
            match_with = match_with,
            resolution = resolution,
            min_likelihood = min_likelihood,
        )
        
    info("ranking velocity genes ...")

    if "spearmans.score" not in adata.var.keys():
        corr = vcorrcoef(
            np.array(adata.layers["ms"]).T,
            np.array(adata.layers["mu"].T),
            mode = "spearmans",
        )
        adata.var["spearmans.score"] = np.clip(corr, 0, None)

    tmp_filter = ~ np.isnan(adata.layers[vkey].sum(0))
    if f"{vkey}.genes" in adata.var.keys():
        tmp_filter &= np.array(adata.var[f"{vkey}.genes"].values, dtype = bool)

    if "unspliced" in adata.layers.keys():
        n_counts = (adata.layers["unspliced"] > 0).sum(0)
        n_counts = n_counts.A1 if issparse(adata.layers["unspliced"]) else n_counts
        min_counts = min(50, np.percentile(n_counts, 50)) if min_counts is None else min_counts
        tmp_filter &= np.ravel(n_counts > min_counts)

    if f"{vkey}.r2" in adata.var.keys():
        r2 = adata.var[f"{vkey}.r2"]
        min_r2 = 0.1 if min_r2 is None else min_r2  # np.percentile(r2[r2 > 0], 50)
        tmp_filter &= r2 > min_r2

    if "spearmans.score" in adata.var.keys():
        corr = adata.var["spearmans.score"]
        min_corr = (
            0.1 if min_corr is None else min_corr
        )  # np.percentile(r2[r2 > 0], 50)
        tmp_filter &= corr > min_corr

    if disp_norm in adata.var.keys():
        dispersions = adata.var[disp_norm]
        min_dispersion = 0 if min_dispersion is None else min_dispersion
        tmp_filter &= dispersions > min_dispersion

    if likelihood in adata.var.keys():
        fit_likelihood = adata.var[likelihood]
        min_likelihood = 0.1 if min_likelihood is None else min_likelihood
        tmp_filter &= fit_likelihood > min_likelihood

    X = adata[:, tmp_filter].layers[vkey]
    groups, groups_masks = select_groups(adata, key=groupby)

    n_groups = groups_masks.shape[0]
    sizes = groups_masks.sum(1)

    mean, var = np.zeros((n_groups, X.shape[1])), np.zeros((n_groups, X.shape[1]))
    for i, mask in enumerate(groups_masks):
        mean[i], var[i] = get_mean_var(X[mask])

    # test each against the union of all other groups
    rankings_gene_names, rankings_gene_scores = [], []
    for i in range(n_groups):
        mask_rest = ~groups_masks[i]
        mean_rest, var_rest = get_mean_var(X[mask_rest])
        size_rest = sizes[i]  # else mask_rest.sum() if method == 't-test'

        scores = (mean[i] - mean_rest) / np.sqrt(
            var[i] / sizes[i] + var_rest / size_rest)
        scores = np.nan_to_num(scores)

        # equivalent to but much faster than np.argsort(scores)[-10:]
        if n_genes > X.shape[1]: n_genes = X.shape[1]
        idx = np.argpartition(scores, -n_genes)[-n_genes:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        rankings_gene_names.append(adata[:, tmp_filter].var_names[idx].values)
        rankings_gene_scores.append(scores[idx])

    rankings_gene_names = np.array([list(n) for n in rankings_gene_names])
    rankings_gene_scores = np.array([list(n) for n in rankings_gene_scores])

    all_names = rankings_gene_names.T.flatten()
    all_scores = rankings_gene_scores.T.flatten()
    vscore = np.zeros(adata.n_vars, dtype=int)
    for i, name in enumerate(adata.var_names):
        if name in all_names: vscore[i] = all_scores[np.where(name == all_names)[0][0]]
    adata.var["velocity.score"] = vscore

    key = "rank.velocity.genes"
    if key not in adata.uns.keys(): adata.uns[key] = {}

    adata.uns[key] = {
        "names": np.rec.fromarrays(
            list(rankings_gene_names), dtype=[(f"{rn}", "U50") for rn in groups]),
        
        "scores": np.rec.fromarrays(
            [n.round(2) for n in rankings_gene_scores],
            dtype=[(f"{rn}", "float32") for rn in groups]),
        
        "params": {
            "groupby": groupby,
            "reference": "rest",
            "method": "t-test_overestim_var",
            "use_raw": True,
        }
    }
