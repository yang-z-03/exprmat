
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, coo_matrix
import matplotlib.pyplot as pl

from exprmat.dynamics.linreg import prod_sum, sum
from exprmat.ansi import error, warning, info
warnings.simplefilter("ignore")


def l2_norm(x, axis: int = 1):

    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    elif x.ndim == 1:
        return np.sqrt(np.einsum("i, i -> ", x, x))
    elif axis == 0:
        return np.sqrt(np.einsum("ij, ij -> j", x, x))
    elif axis == 1:
        return np.sqrt(np.einsum("ij, ij -> i", x, x))


def round(k, dec = 2, as_str = None):
    
    if isinstance(k, (list, tuple, np.record, np.ndarray)):
        return [round(ki, dec) for ki in k]
    
    if "e" in f"{k}":
        k_str = f"{k}".split("e")
        result = f"{np.round(float(k_str[0]), dec)}1e{k_str[1]}"
        return f"{result}" if as_str else float(result)
    
    result = np.round(float(k), dec)
    return f"{result}" if as_str else result


def mean(x, axis=0):
    return x.mean(axis).A1 if issparse(x) else x.mean(axis)


def make_dense(X):
    XA = X.toarray() if issparse(X) and X.ndim == 2 else X.A1 if issparse(X) else X
    if XA.ndim == 2:
        XA = XA[0] if XA.shape[0] == 1 else XA[:, 0] if XA.shape[1] == 1 else XA
    return np.array(XA)


def r_square(residual, total):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = np.ones(residual.shape[1]) - prod_sum(
            residual, residual, axis = 0
        ) / prod_sum(total, total, axis = 0)
    r2[np.isnan(r2)] = 0
    return r2


def cosine_correlation(dX, Vi):
    
    dx = dX - dX.mean(-1)[:, None]
    Vi_norm = l2_norm(Vi, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Vi_norm == 0: result = np.zeros(dx.shape[0])
        else: result = np.einsum("ij, j", dx, Vi) / (l2_norm(dx, axis=1) * Vi_norm)[None, :]

    return result


def normalize(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if issparse(X): return X.multiply(csr_matrix(1.0 / np.abs(X).sum(1)))
        else: return X / X.sum(1)


def scale(X, min = 0, max = 1):

    idx = np.isfinite(X)
    if any(idx):
        X = X - X[idx].min() + min
        xmax = X[idx].max()
        X = X / xmax * max if xmax != 0 else X * max
    
    return X


def get_neighs(adata, neighbor_key = 'neighbors', mode = "distances"):

    if neighbor_key in adata.uns.keys() and f'{mode}_key' in adata.uns[neighbor_key]:
        return adata.obsp[adata.uns[neighbor_key][f'{mode}_key']]
    elif hasattr(adata, "obsp") and mode in adata.obsp.keys():
        return adata.obsp[mode]
    else: error("the selected mode is not valid.")


def get_n_neighs(adata, neighbor_key = 'neighbors'):
    return adata.uns.get(neighbor_key, {}).get("n_neighbors", 0)


def get_connectivities(
    adata, neighbor_key = 'neighbors', mode = "connectivities", 
    n_neighbors = None, recurse_neighbors = False
):
    if neighbor_key in adata.uns.keys():
        C = get_neighs(adata, neighbor_key, mode)
        if n_neighbors is not None and n_neighbors < get_n_neighs(adata):
            if mode == "connectivities":
                C = select_connectivities(C, n_neighbors)
            else: C = select_distances(C, n_neighbors)
        
        connectivities = C > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            connectivities.setdiag(1)
            if recurse_neighbors:
                connectivities += connectivities.dot(connectivities * 0.5)
                connectivities.data = np.clip(connectivities.data, 0, 1)
            connectivities = connectivities.multiply(1.0 / connectivities.sum(1))
        return connectivities.tocsr().astype(np.float32)
    else: return None


def select_distances(dist, n_neighbors = None):

    D = dist.copy()
    n_counts = (D > 0).sum(1).A1 if issparse(D) else (D > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None 
        else min(n_counts.min(), n_neighbors)
    )

    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0

    D.eliminate_zeros()
    return D


def select_connectivities(connectivities, n_neighbors = None):

    C = connectivities.copy()
    n_counts = (C > 0).sum(1).A1 if issparse(C) else (C > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None 
        else min(n_counts.min(), n_neighbors)
    )

    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = C.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[::-1][n_neighbors:]
        dat[rm_idx] = 0

    C.eliminate_zeros()
    return C


def get_csr_from_indices(knn_indices, knn_dists, n_obs, n_neighbors):

    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1: continue  # we didn't get the full knn for i
            if knn_indices[i, j] == i: val = 0.0
            else: val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


# TODO: Finish docstrings
def compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs, n_neighbors,
    set_op_mix_ratio = 1.0,
    local_connectivity = 1.0,
):
    """
    Computes fuzzy simplical set associated with data.

    This is from umap.fuzzy_simplicial_set :cite:p:`McInnes18`. Given a set of data X, 
    a neighborhood size, and a measure of distance compute the fuzzy simplicial set 
    (here represented as a fuzzy graph in the form of a sparse matrix) associated to 
    the data. This is done by locally approximating geodesic distance at each point, 
    creating a fuzzy simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors, None, None,
        knn_indices = knn_indices,
        knn_dists = knn_dists,
        set_op_mix_ratio = set_op_mix_ratio,
        local_connectivity = local_connectivity
    )

    if isinstance(connectivities, tuple):  # umap returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = get_csr_from_indices(knn_indices, knn_dists, n_obs, n_neighbors)
    return distances, connectivities.tocsr()


def get_indices(dist, n_neighbors = None, mode_neighbors = "distances"):

    D = dist.copy()
    D.data += 1e-6

    n_counts = sum(D > 0, axis=1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None 
        else min(n_counts.min(), n_neighbors)
    )

    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0

    D.eliminate_zeros()
    D.data -= 1e-6

    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))

    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors)
        indices = get_indices_from_csr(conn)
    
    return indices, D


def get_indices_from_csr(conn):

    # extracts indices from connectivity matrix, pads with nans
    ixs = np.ones((conn.shape[0], np.max((conn > 0).sum(1)))) * np.nan
    for i in range(ixs.shape[0]):
        cell_indices = conn[i, :].indices
        ixs[i, : len(cell_indices)] = cell_indices
    return ixs


def get_iterative_indices(
    indices, index, n_recurse_neighbors = 2,
    max_neighs = None,
):
    
    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices


def geometric_matrix_sum(C, n_power = 2):  
    
    C_n = ( # computes C + C^2 + C^3 + ...
        geometric_matrix_sum(C, n_power - 1) if n_power > 2 
        else C if n_power > 1 
        else 0
    )

    return C + C.dot(C_n)


def groups_to_bool(adata, groups, groupby = None):

    groups = [groups] if isinstance(groups, str) else groups
    if isinstance(groups, (list, tuple, np.ndarray, np.record)):
        
        groupby = (
            groupby
            if groupby in adata.obs.keys()
            else "clusters"
            if "clusters" in adata.obs.keys()
            else "louvain"
            if "louvain" in adata.obs.keys()
            else "leiden"
            if "leiden" in adata.obs.keys()
            else None
        )
        if groupby is not None:
            groups = np.array([key in groups for key in adata.obs[groupby]])
        else: error("groupby attribute not valid.")
    
    return groups


def most_common_in_list(lst):

    lst = [item for item in lst if item is not np.nan and item != "nan"]
    lst = list(lst)
    return max(set(lst), key = lst.count)


def randomized_velocity(adata, vkey = "velocity", add_key = "velocity.random"):

    v_rand = adata.layers[vkey].copy()
    for i in range(v_rand.shape[1]):
        np.random.shuffle(v_rand[:, i])
        v_rand[:, i] = v_rand[:, i] * np.random.choice(
            np.array([+1, -1]), size = v_rand.shape[0])
    adata.layers[add_key] = v_rand

    from exprmat.dynamics.velocity import velocity_embedding
    from exprmat.dynamics.velocity import velocity_graph

    velocity_graph(adata, vkey = add_key)
    velocity_embedding(adata, vkey = add_key, autoscale=False)


def extract_int_from_str(array):

    def str_to_int(item):
        num = "".join(filter(str.isdigit, item))
        num = int(num) if len(num) > 0 else -1
        return num

    if isinstance(array, str):
        nums = str_to_int(array)
    
    elif len(array) > 1 and isinstance(array[0], str):
        nums = []
        for item in array: nums.append(str_to_int(item))
    else: nums = array

    nums = pd.Categorical(nums) if array.dtype == "category" else np.array(nums)
    return nums


def strings_to_categoricals(adata):
    
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or 
            is_integer_dtype(values) or 
            is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100): df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100): df[key] = c


def merge_groups(adata, key, map_groups, key_added = None, map_colors = None):

    strings_to_categoricals(adata)
    if len(map_groups) != len(adata.obs[key].cat.categories):
        map_coarse = {}
        for c in adata.obs[key].cat.categories:
            for group in map_groups:
                if any(cluster == c for cluster in map_groups[group]):
                    map_coarse[c] = group
            if c not in map_coarse: map_coarse[c] = c
        map_groups = map_coarse

    if key_added is None: key_added = f"{key}.coarse"

    from pandas.api.types import CategoricalDtype
    adata.obs[key_added] = adata.obs[key].map(map_groups).astype(CategoricalDtype())
    old_categories = adata.obs[key].cat.categories
    new_categories = adata.obs[key_added].cat.categories

    # map_colors is passed
    if map_colors is not None:
        old_colors = None
        if f"{key}.colors" in adata.uns:
            old_colors = adata.uns[f"{key}.colors"]
        new_colors = []
        for group in adata.obs[key_added].cat.categories:
            if group in map_colors:
                new_colors.append(map_colors[group])
            elif group in old_categories and old_colors is not None:
                new_colors.append(old_colors[old_categories.get_loc(group)])
            else: error(f"didn't specify a color for {group}.")
        adata.uns[f"{key_added}.colors"] = new_colors

    # map_colors is not passed
    elif f"{key}.colors" in adata.uns:
        old_colors = adata.uns[f"{key}.colors"]
        inverse_map_groups = {g: [] for g in new_categories}
        for old_group in old_categories:
            inverse_map_groups[map_groups[old_group]].append(old_group)
        new_colors = []
        for group in new_categories:
            # take the largest of the old groups
            old_group = (
                adata.obs[key][adata.obs[key].isin(inverse_map_groups[group])]
                .value_counts().index[0])
            new_colors.append(old_colors[old_categories.get_loc(old_group)])
        adata.uns[f"{key_added}.colors"] = new_colors


def cutoff_small_velocities(
    adata, vkey = "velocity", key_added = "velocity.cut", frac_of_max = 0.5, use_raw = False
):
    
    x = adata.layers["spliced"] if use_raw else adata.layers["ms"]
    y = adata.layers["unspliced"] if use_raw else adata.layers["mu"]

    x_max = x.max(0).toarray()[0] if issparse(x) else x.max(0)
    y_max = y.max(0).toarray()[0] if issparse(y) else y.max(0)

    xy_norm = x / np.clip(x_max, 1e-3, None) + y / np.clip(y_max, 1e-3, None)
    W = xy_norm >= np.percentile(xy_norm, 98, axis=0) * frac_of_max

    adata.layers[key_added] = csr_matrix(W).multiply(adata.layers[vkey]).tocsr()

    from exprmat.dynamics.velocity import velocity_embedding
    from exprmat.dynamics.velocity import velocity_graph

    velocity_graph(adata, vkey = key_added, approx=True)
    velocity_embedding(adata, vkey = key_added)


def make_unique_list(key, allow_array=False):

    from pandas import Index, unique

    if isinstance(key, Index): key = key.tolist()

    is_list = (
        isinstance(key, (list, tuple, np.record))
        if allow_array
        else isinstance(key, (list, tuple, np.ndarray, np.record))
    )

    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return unique(key) if is_list_of_str else key if is_list and len(key) < 20 else [key]


def test_bimodality(x, bins = 30, kde = True, plot = False):
    """ Test for bimodal distribution. """

    from scipy.stats import gaussian_kde, norm
    lb, ub = np.min(x), np.percentile(x, 99.9)
    grid = np.linspace(lb, ub if ub <= lb else np.max(x), bins)
    kde_grid = gaussian_kde(x)(grid) if kde else \
        np.histogram(x, bins=grid, density=True)[0]

    idx = int(bins / 2) - 2
    end = idx + 4
    idx += np.argmin(kde_grid[idx:end])

    peak_0 = kde_grid[:idx].argmax()
    peak_1 = kde_grid[idx:].argmax()
    kde_peak = kde_grid[idx:][peak_1]
    kde_mid = kde_grid[idx:].mean()

    t_stat = (kde_peak - kde_mid) / np.clip(np.std(kde_grid) / np.sqrt(bins), 1, None)
    p_val = norm.sf(t_stat)

    grid_0 = grid[:idx]
    grid_1 = grid[idx:]
    means = [
        (grid_0[peak_0] + grid_0[min(peak_0 + 1, len(grid_0) - 1)]) / 2,
        (grid_1[peak_1] + grid_1[min(peak_1 + 1, len(grid_1) - 1)]) / 2,
    ]

    if plot:
        
        color = "grey"
        if kde:
            pl.plot(grid, kde_grid, color = color)
            pl.fill_between(grid, 0, kde_grid, alpha = 0.4, color = color)
        else: pl.hist(x, bins = grid, alpha = 0.4, density = True, color = color)
        
        pl.axvline(means[0], color = color)
        pl.axvline(means[1], color = color)
        pl.axhline(kde_mid, alpha = 0.2, linestyle = "--", color = color)
        pl.show()

    # ~ t_test (reject unimodality if t_stat > 3)
    return t_stat, p_val, means


def random_subsample(adata, fraction=0.1, return_subset=False, copy=False):

    adata_sub = adata.copy() if copy else adata
    p, size = fraction, adata.n_obs
    subset = np.random.choice([True, False], size = size, p = [p, 1 - p])
    adata_sub._inplace_subset_obs(subset)
    return adata_sub if copy else subset if return_subset else None


def get_duplicates(array):
    from collections import Counter
    return np.array([item for (item, count) in Counter(array).items() if count > 1])


def corrcoef(x, y, mode = "pearsons"):
    from scipy.stats import pearsonr, spearmanr
    corr, _ = spearmanr(x, y) if mode == "spearmans" else pearsonr(x, y)
    return corr


def vcorrcoef(X, y, mode = "pearsons", axis = -1):
    """
    Pearsons/Spearmans correlation coefficients.
    Use Pearsons / Spearmans to test for linear / monotonic relationship.

    Parameters
    ----------
    X: np.ndarray
        Data vector or matrix

    y: np.ndarray
        Data vector or matrix

    mode: 'pearsons' or 'spearmans' (default: 'pearsons')
        Which correlation metric to use.
    """

    if issparse(X): X = np.array(X.toarray())
    if issparse(y): y = np.array(y.toarray())
    if axis == 0:
        if X.ndim > 1: X = np.array(X.T)
        if y.ndim > 1: y = np.array(y.T)

    if X.shape[axis] != y.shape[axis]: X = X.T
    if mode in {"spearmans", "spearman"}:
        from scipy.stats.stats import rankdata
        X = np.apply_along_axis(rankdata, axis=-1, arr=X)
        y = np.apply_along_axis(rankdata, axis=-1, arr=y)

    Xm = np.array(X - (np.nanmean(X, -1)[:, None] if X.ndim > 1 else np.nanmean(X, -1)))
    ym = np.array(y - (np.nanmean(y, -1)[:, None] if y.ndim > 1 else np.nanmean(y, -1)))
    corr = np.nansum(Xm * ym, -1) / np.sqrt(
        np.nansum(Xm ** 2, -1) * np.nansum(ym ** 2, -1))
    return corr


def isin(x, y): return np.array(pd.DataFrame(x).isin(y)).flatten()

def indices_to_bool(indices, n): return isin(np.arange(n), indices)


def convolve(
    adata, x, neighbor_key: str = 'neighbors',
    mode: str = "connectivities"
):
    conn = get_connectivities(adata, neighbor_key, mode)
    if isinstance(x, str) and x in adata.layers.keys(): x = adata.layers[x]
    if x.ndim == 1: return conn.dot(x)
    idx_valid = ~ np.isnan(x.sum(0))
    Y = np.ones(x.shape) * np.nan
    Y[:, idx_valid] = conn.dot(x[:, idx_valid])
    return Y


def get_extrapolated_state(adata, vkey = "velocity", dt=1, use_raw = None, dropna = True):
    """ Get extrapolated cell state. """
    S = adata.layers["spliced" if use_raw else "ms"]
    if dropna:
        St = S + dt * adata.layers[vkey]
        St = St[:, np.isfinite(np.sum(St, 0))]
    else: St = S + dt * np.nan_to_num(adata.layers[vkey])
    return St


def get_plasticity_score(adata, modality = "ms"):

    idx_top_genes = np.argsort(adata.var["gene.count.corr"].values)[::-1][:200]
    Ms = np.array(adata.layers[modality][:, idx_top_genes])
    return scale(np.mean(Ms / np.max(Ms, axis = 0), axis = 1))