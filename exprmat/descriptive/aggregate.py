'''
Summary accepts two to three categoricals in the obs metadata to reform the dataset
with summarizing statistics. Aggregates takes one obs category and one var category
to aggregate observations and variables.
'''

from __future__ import annotations

import numpy as np
import anndata as ad
from typing import Literal
import logging
from pathlib import Path
import numpy as np
import functools

from exprmat.utils import choose_layer
import exprmat.snapatac as internal
from exprmat.utils import is_anndata 
from exprmat.preprocessing.linear import get_size_factor_for_regions, normalize_size_factor, rpm
from exprmat.ansi import error, warning, info


def aggregate(
    adata, data = 'X', method = 'mean', method_args = {},
    obs_key = 'sample', var_key = None
):
    source = choose_layer(adata, layer = data)

    # TODO: maybe some alternative method rather than silly iteration :(

    if (obs_key is None) and (var_key is not None):
        return aggregate_var(
            adata, data = data, method = method, method_args = method_args,
            var_key = var_key
        )
    
    if (var_key is None) and (obs_key is not None):
        return aggregate_obs(
            adata, data = data, method = method, method_args = method_args,
            obs_key = obs_key
        )
    
    obs_val = adata.obs_names.tolist()
    obs = adata.obs_names.to_numpy()
    if obs_key is not None:
        obs_val = adata.obs[obs_key].unique().tolist()
        obs = adata.obs[obs_key].to_numpy()
    
    var_val = adata.var_names.tolist()
    var = adata.var_names.to_numpy()
    if var_key is not None:
        var_val = adata.var[var_key].unique().tolist()
        var = adata.var[var_key].to_numpy()
    
    mat = np.ndarray(shape = (len(obs_val), len(var_val)), dtype = np.float32)
    meth = None

    methods = {
        'mean': np.mean
    }

    for i in range(len(obs_val)):
        for j in range(len(var_val)):
            
            mat[i, j] = methods[method](
                source[obs == obs_val[i], var == var_val[j]], 
                **method_args
            )
    
    annd = ad.AnnData(X = mat)
    annd.obs_names = obs_val
    annd.var_names = var_val
    return annd


def aggregate_obs(
    adata, data = 'X', method = 'mean', method_args = {}, obs_key = 'sample'
):
    
    source = choose_layer(adata, layer = data)

    obs_val = adata.obs_names.tolist()
    obs = adata.obs_names.to_numpy()
    if obs_key is not None:
        obs_val = adata.obs[obs_key].unique().tolist()
        obs = adata.obs[obs_key].to_numpy()
    
    mat = np.ndarray(shape = (len(obs_val), adata.n_vars), dtype = np.float32)

    methods = {
        'mean': np.mean
    }

    for i in range(len(obs_val)):
        mat[i, :] = methods[method](
            source[obs == obs_val[i], :],
            axis = 0,
            **method_args
        )
    
    annd = ad.AnnData(X = mat)
    annd.obs_names = obs_val
    annd.var_names = adata.var_names.tolist()
    annd.var = adata.var.copy()
    return annd


def aggregate_var(
    adata, data = 'X', method = 'mean', method_args = {}, var_key = 'module'
):
    
    source = choose_layer(adata, layer = data)

    var_val = adata.var_names.tolist()
    var = adata.var_names.to_numpy()
    if var_key is not None:
        var_val = adata.var[var_key].unique().tolist()
        var = adata.var[var_key].to_numpy()
    
    mat = np.ndarray(shape = (adata.n_obs, len(var_val)), dtype = np.float32)

    methods = {
        'mean': np.mean
    }

    for i in range(len(var_val)):
        mat[:, i] = methods[method](
            source[:, var == var_val[i]],
            axis = 1, # row sums
            **method_args
        )
    
    annd = ad.AnnData(X = mat)
    annd.obs_names = adata.obs_names.tolist()
    annd.obs = adata.obs.copy()
    annd.var_names = var_val
    return annd


def aggregate_groups(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str] | None = None,
    normalize: Literal["rpm", "rpkm"] | None = None,
    file: Path | None = None,
) -> np.ndarray | internal.AnnData:
    """
    Aggregate values in adata.X in a row-wise fashion.

    Aggregate values in adata.X in a row-wise fashion. This is used to compute
    RPKM or RPM values stratified by user-provided groupings.

    Parameters
    ----------
    adata
        The AnnData or AnnDataSet object.
    
    groupby
        Group the cells into different groups. If a `str`, groups are obtained
        from `.obs[groupby]`.

    normalize
        normalization method: "RPM" or "RPKM". For 10x data with UMIs, there is
        no sequencing bias on longer fragments. So you should not use RPKM for them.
    """

    from natsort import natsorted
    from anndata import AnnData

    def norm(x):
        if normalize is None: return x
        elif normalize == "rpkm":
            size_factor = get_size_factor_for_regions(adata.var_names) / 1000.0
            return normalize_size_factor(x, size_factor)
        elif normalize == "RPM":
            return rpm(x)
        else: error("mormalization method must be 'rpkm' or 'rpm'")

    if groupby is None:
        row_sum = functools.reduce(
            lambda a, b: a + b,
            (np.ravel(chunk.sum(axis=0)) for chunk, _, _ in adata.chunked_X(1000)),
        )
        row_sum = norm(row_sum)
        return row_sum
    
    else:
        groups = adata.obs[groupby].to_numpy() if isinstance(groupby, str) else np.array(groupby)
        if groups.size != adata.n_obs:
            error("the length of `groupby` should equal to the number of obervations")

        result = {x: np.zeros(adata.n_vars) for x in natsorted(np.unique(groups))}
        for chunk, start, stop in adata.chunked_X(2000):
            for i in range(start, stop): result[groups[i]] += chunk[i-start, :]
        for k in result.keys(): result[k] = norm(np.ravel(result[k]))

        keys, values = zip(*result.items())
        if file is None: out_adata = AnnData(X = np.array(values))
        else: out_adata = internal.AnnData(filename = file, X = np.array(values))
        out_adata.obs_names = list(keys)
        out_adata.var_names = adata.var_names
        return out_adata
    

def aggregate_cells(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    use_rep: str = 'spectral',
    target_num_cells: int | None = None,
    min_cluster_size: int = 50,
    random_state: int = 0,
    key_added: str = 'metacells',
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Aggregate cells into pseudo-cells by iterative clustering

    Parameters
    ----------
    adata
        AnnData or AnnDataSet object or matrix.

    use_rep
        `adata.obs` key for retrieving the input similarity matrix.

    target_num_cells
        If None, `target_num_cells = num_cells / min_cluster_size`.

    min_cluster_size
        The minimum size of clusters.
    """

    def clustering(data):
        from exprmat.reduction.nn import knn_graph
        from exprmat.clustering.leiden import leiden
        from anndata import AnnData
        _, _, dist_mat = knn_graph(data, k = 50, approx = True)
        adata = AnnData(X = data)
        leiden(adata, adjacency = dist_mat, resolution = 1, key_added = '.leiden', random_state = random_state)
        result = adata.obs['.leiden'].copy()
        del adata
        return result

    if is_anndata(adata):
        X = adata.obsm[use_rep]
    else:
        inplace = False
        X = adata

    if target_num_cells is None:
        target_num_cells = X.shape[0] // min_cluster_size

    info("performing initial clustering ...")
    membership = clustering(X).astype('object')
    cluster_ids = [x for x in np.unique(membership) if x != "-1"]
    ids_next = cluster_ids
    n_clusters = len(cluster_ids)
    depth = 0

    while n_clusters < target_num_cells and len(ids_next) > 0:
        depth += 1
        logging.info("iterative clustering: {}, number of clusters: {}".format(depth, n_clusters))
        ids = set()
        for cid in ids_next:
            mask = membership == cid
            sub_clusters = clustering(X[mask, :])
            n_sub_clusters = np.count_nonzero(np.unique(sub_clusters) != "-1")
            if n_sub_clusters > 1 and np.count_nonzero(sub_clusters != "-1") / sub_clusters.shape[0] > 0.9:
                n_clusters += n_sub_clusters - 1
                for i, i_ in enumerate(np.where(mask)[0]):
                    lab = sub_clusters[i]
                    if lab == "-1": membership[i_] = lab
                    else:
                        new_lab = membership[i_] + "." + lab
                        membership[i_] = new_lab
                        ids.add(new_lab)

            if n_clusters >= target_num_cells: break
        ids_next = ids

    info("asked for {} pseudo-cells. got: {}.".format(target_num_cells, n_clusters))

    if inplace:
        import polars
        adata.obs[key_added] = polars.Series(
            [str(x) for x in membership],
            dtype = polars.datatypes.Categorical,
        )

    else: return membership
