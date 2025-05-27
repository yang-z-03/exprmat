
from __future__ import annotations
from numpy import log10, finfo, exp
from numpy import union1d
import anndata
import numpy as np
from tqdm import tqdm

from joblib import Parallel, delayed
from exprmat.ansi import warning, error


class default_params():
    
    logbase = exp(1)
    min_cells = 5
    expr_prop = 0.1
    # set to a smaller value. lianapy uses 1000 by default.
    n_perms = 500
    seed = 42
    de_method = 't-test'
    resource_name = 'consensus'
    resource = None
    interactions = None
    layer = None
    use_raw = True
    verbose = False
    return_all_lrs = False
    supp_columns = None
    inplace = True
    groupby_pairs = None
    return_fig = True
    cmap = 'turbo'
    lr_sep = '^'
    complex_sep = "_"

    def inverse_fun(x):
        return - log10(x + finfo(float).eps)

class default_anndata_keys():
    uns_key = 'lr'
    spatial_key = 'spatial'
    connectivity_key = f'{spatial_key}.connectivities'
    target_metrics = 'target.metrics'
    interactions = 'interactions'

class default_primary_columns():
    source = 'source'
    target = 'target'
    ligand = 'ligand'
    receptor = 'receptor'
    ligand_complex = 'ligand.complex'
    receptor_complex = 'receptor.complex'
    primary = [source, target, ligand_complex, receptor_complex]
    complete = primary + [ligand, receptor]

class default_common_columns():
    ligand_means = 'ligand.means'
    receptor_means = 'receptor.means'
    ligand_props = 'ligand.props'
    receptor_props = 'receptor.props'
    ligand_pvals = 'ligand.p'
    receptor_pvals = 'receptor.p'

class common_method_columns():
    ligand_means_sums = 'ligand.means.sums'
    receptor_means_sums = 'receptor.means.sums'
    ligand_zscores = 'ligand.zscores'
    receptor_zscores = 'receptor.zscores'
    ligand_logfc = 'ligand.logfc'
    receptor_logfc = 'receptor.logfc'
    ligand_trimean = 'ligand.trimean'
    receptor_trimean = 'receptor.trimean'
    mat_mean = 'mat.mean'
    mat_max = 'mat.max'
    ligand_cdf = 'ligand.cdf'
    receptor_cdf = 'receptor.cdf'

    @classmethod
    def get_all_values(cls):
        return [value for name, value in cls.__dict__.items()
                if not name.startswith('__') and isinstance(value, str)]

class internal_values():
    lrs_to_keep = '.lrs.keep'
    prop_min = '.min.prop'
    label = '.label'


def join_stats(source, target, dedict, resource):

    source_stats = dedict[source].copy()
    source_stats.columns = source_stats.columns.map(
        lambda x: 'ligand.' + str(x))
    source_stats = source_stats.rename(
        columns={'ligand.names': 'ligand', 'ligand.label': 'source'})

    target_stats = dedict[target].copy()
    target_stats.columns = target_stats.columns.map(
        lambda x: 'receptor.' + str(x))
    target_stats = target_stats.rename(
        columns={'receptor.names': 'receptor', 'receptor.label': 'target'})

    bound = resource.merge(source_stats).merge(target_stats)

    return bound


def get_props(X):
    return X.getnnz(axis = 0) / X.shape[0]

def get_groupby_subset(groupby_pairs):

    if groupby_pairs is not default_params.groupby_pairs:
        if not (default_primary_columns.source in groupby_pairs.columns) | \
            (default_primary_columns.target in groupby_pairs.columns):

            error(
                f"{default_primary_columns.source} and {default_primary_columns.target} "
                f"must be in groupby_pairs"
            )
        
        groupby_subset = union1d(
            groupby_pairs[default_primary_columns.source].unique(), 
            groupby_pairs[default_primary_columns.target].unique()
        )

    else: groupby_subset = None
    return groupby_subset


def get_means_permutations(
    adata: anndata.AnnData,
    n_perms: int,
    seed: int,
    agg_fun,
    norm_factor: float | None,
    n_jobs: int,
    verbose: bool
):
    """
    Generate permutations and indices required for permutation-based methods

    Parameters
    ----------
    adata
        Annotated data matrix
    
    n_perms
        Number of permutations to be calculated
    
    seed
        Random seed for reproducibility.
    
    agg_fun
        Function by which to aggregate the matrix, should take `axis` argument
    
    norm_factor
        Additionally normalize the data by some factor (e.g. matrix max for CellChat)

    Returns
    -------
    Tuple with:
        - perms: 3D tensor with permuted averages per cluster
        - ligand_pos: Index of the ligand in the tensor
        - receptor_pos: Index of the receptor in the perms tensor
        - labels_pos: Index of cell identities in the perms tensor
    """

    if isinstance(norm_factor, np.float32):
        adata.X /= norm_factor

    # define labels and masks
    labels = adata.obs[internal_values.label].cat.categories
    labels_mask = np.zeros((adata.shape[0], labels.shape[0]), dtype = bool)

    # populate masks shape(genes, labels)
    for ct_idx, label in enumerate(labels):
        labels_mask[:, ct_idx] = adata.obs[internal_values.label] == label

    # perm should be a cube /w dims: n_perms x idents x n_genes
    perms = generate_perms_cube(
        adata.X, n_perms, labels_mask, seed, agg_fun, n_jobs, verbose = verbose)

    return perms


# Define a helper function for parallel processing
def permute_and_aggregate(perm, perm_idx, X, labels_mask, agg_fun):

    perm_mat = X[perm_idx]
    permuted_means = np.array([
        agg_fun(perm_mat[labels_mask[:, i]], axis = 0) 
        for i in range(labels_mask.shape[1])
    ])

    return perm, permuted_means


def generate_perms_cube(X, n_perms, labels_mask, seed, agg_fun, n_jobs, verbose):
    # initialize rng
    rng = np.random.default_rng(seed=seed)

    # indexes to be shuffled
    idx = np.arange(X.shape[0])

    # perm should be a cube /w dims: n_perms x idents x n_genes
    perms = np.zeros((n_perms, labels_mask.shape[1], X.shape[1]))

    # use Parallel to enable parallelization
    results = Parallel(n_jobs = n_jobs)(
        delayed(permute_and_aggregate)
        (perm, rng.permutation(idx), X, labels_mask, agg_fun)
        for perm in tqdm(range(n_perms), disable = not verbose, ncols = 80)
    )

    # unpack results
    for perm, permuted_means in results:
        perms[perm] = np.reshape(permuted_means, (labels_mask.shape[1], X.shape[1]))

    return perms


def get_positions(adata, lr_res):
    
    labels = adata.obs[internal_values.label].cat.categories

    # get positions of each entity in the matrix
    ligand_pos = {
        entity: np.where(adata.var_names == entity)[0][0] 
        for entity in lr_res[default_primary_columns.ligand]
    }
    receptor_pos = {
        entity: np.where(adata.var_names == entity)[0][0] 
        for entity in lr_res[default_primary_columns.receptor]
    }
    labels_pos = { labels[pos]: pos for pos in range(labels.shape[0]) }

    return ligand_pos, receptor_pos, labels_pos


def get_mat_idx(adata, lr_res):

    # convert to indexes
    ligand_pos, receptor_pos, labels_pos = get_positions(adata, lr_res)

    ligand_idx = lr_res[default_primary_columns.ligand].map(ligand_pos)
    receptor_idx = lr_res[default_primary_columns.receptor].map(receptor_pos)

    source_idx = lr_res[default_primary_columns.source].map(labels_pos)
    target_idx = lr_res[default_primary_columns.target].map(labels_pos)

    return ligand_idx, receptor_idx, source_idx, target_idx


def calculate_pvals(lr_truth, perm_stats, _score_fun):

    # calculate p-values
    if perm_stats is not None:
        lr_perm_means = _score_fun(perm_stats, axis=0)
        n_perms = perm_stats.shape[1]
        pvals = np.sum(np.greater_equal(lr_perm_means, lr_truth), axis = 0) / n_perms
    
    else: pvals = None
    return pvals