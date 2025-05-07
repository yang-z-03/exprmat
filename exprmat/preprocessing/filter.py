
import numpy as np
import scipy
import scipy.sparse as sparse
import pandas as pd
import numba
from numba import njit

from exprmat.ansi import warning, info, error


def running_quantile(x, y, p, n_bins = 20):
    '''
    Calculate the quantiles of y (on given cutoff probability p) running over sub-ranges
    of x. This gives the quantile ranges of y that is correlated with x.
    '''

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / n_bins
    # segment x axis into bins
    x_out = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, n_bins)

    y_out = np.zeros(x_out.shape)
    for i in range(len(x_out)):
        ind = np.nonzero((x >= x_out[i] - dx / 2) & (x < x_out[i] + dx / 2))[0]
        if len(ind) > 0: # if this segment contains values
            y_out[i] = np.percentile(y[ind], p)
        else:
            if i > 0: y_out[i] = y_out[i - 1]
            else: y_out[i] = np.nan

    return x_out, y_out


def vscore(E, min_mean = 0, n_bins = 50, fit_percentile = 0.1, error_weight = 1):
    '''
    Calculate v-score (above poisson noise statistic) for genes in the input counts matrix
    
    Returns
    ----------

    A tuple containing the following components:

    *   V-scores for each gene.
    *   Coefficient of variation (CV) for effective counts.
    *   Coefficient of variation (CV) for input counts.
    *   Indices of genes considered in the analysis, namely genes that pass the minimal
        expression level filter. (given by ``min_mean``). Only expressing genes are
        candidates of variable gene selection.
    *   Mean expression values for selected genes.
    *   Fano factor (variance to mean ratio) for selected genes.
    *   Parameter 'a' obtained during optimization. (fitting y = log(c * exp(x)) + b)
    *   Parameter 'b' obtained during optimization.
    '''

    n_cell = E.shape[0]

    mu_gene = E.mean(axis = 0).A.squeeze() # mean expression of genes
    min_expr_filter = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[min_expr_filter]

    tmp = E[:, min_expr_filter]
    tmp.data **= 2
    var_gene = tmp.mean(axis = 0).A.squeeze() - mu_gene ** 2
    ff_gene = var_gene / mu_gene # signal to noise ratio, fano factor
    del tmp

    data_x = np.log(mu_gene)
    data_y = np.log(ff_gene / mu_gene)

    # this plots the expression of genes as x, and the variance of genes as y
    # and both highly-variable genes and stable housekeeping genes selection can
    # be performed on this expression - variance plot using running quantile.
    x, y = running_quantile(data_x, data_y, fit_percentile, n_bins)
    x = x[~ np.isnan(y)]
    y = y[~ np.isnan(y)]

    generalize_log = lambda input: np.log(input[1] * np.exp(- input[0]) + input[2])
    h, b = np.histogram(np.log(ff_gene[mu_gene > 0]), bins = 200)
    b = b[:-1] + np.diff(b) / 2
    max_index = np.argmax(h)
    c = np.max((np.exp(b[max_index]), 1))
    # fit y = log(c * exp(x)) + b
    error_func = lambda b2: np.sum(abs(generalize_log([x, c, b2]) - y) ** error_weight)
    initial_b = 0.1
    b = scipy.optimize.fmin(func = error_func, x0 = [initial_b], disp = False)
    a = c / (1 + b) - 1

    v_scores = ff_gene / ((1 + a) * (1 + b) + b * mu_gene)
    cv_eff = np.sqrt((1 + a) * (1 + b) - 1)
    cv_input = np.sqrt(b)

    return v_scores, cv_eff, cv_input, min_expr_filter, mu_gene, ff_gene, a, b


def highly_variable_genes_ff(E, cells_id = [], min_vscore_pct = 85, min_counts = 3, min_cells = 3):
    ''' 
    Filter highly variable genes by fano factor threshold

    Parameters
    -----------

    E : np.array | scipy.sparse.csc_matrix | scipy.sparse.csr_matrix
        numpy array of feature counts matrix, either dense or sparse.

    cells_id : List
        Select a partition of cells to perform highly variable selection
    
    min_vscore_pct : int
        Filter the top (100 - min_vscore_pct) signal to noise ratio as highly variable

    min_counts : int
    min_cells : int
        Should only keep genes with at least ``min_counts`` transcripts detected in at
        least ``min_cells`` cells. Otherwise the gene is treated as lean.
    '''

    if len(cells_id) == 0: cells_id = np.arange(E.shape[0])

    vscores, _, _, gene_id, mu_gene, fano_factors, _, _ = vscore(E[cells_id, :])
    ix2 = vscores > 0
    vscores = vscores[ix2]
    gene_id = gene_id[ix2]
    mu_gene = mu_gene[ix2]
    fano_factors = fano_factors[ix2]
    min_vscore = np.percentile(vscores, min_vscore_pct)

    selector = (
        # only keep genes expressing at >= min_counts level in at least min_cells.
        ((E[:,gene_id] >= min_counts).sum(0).A.squeeze() >= min_cells) & \
        # select highly variable genes
        (vscores >= min_vscore)
    )
    
    return gene_id[selector]


def axis_mean(X: np.ndarray, *, axis, dtype) -> np.ndarray:
    return X.mean(axis = axis, dtype = dtype)


def elem_mul(x, y):
    if isinstance(x, sparse.csc_matrix) or isinstance(x, sparse.csr_matrix):
        # returns coo_matrix, so cast back to input type
        return type(x)(x.multiply(y))
    return x * y


def mean_variance(X, axis = 0):

    if isinstance(X, sparse.csr_matrix) or isinstance(X, sparse.csc_matrix):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = axis_mean(X, axis = axis, dtype=np.float64)
        mean_sq = axis_mean(elem_mul(X, X), axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    if X.shape[axis] != 1:
        var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def sparse_mean_variance_axis(mtx, axis):
    """
    Compute mean and variance along one axis of a sparse matrix.
    This code and internal functions are based on sklearns `sparsefuncs.mean_variance_axis`.

    Modifications:
    *   Allow deciding on the output type, which can increase accuracy when calculating 
        the mean and variance of 32-bit floats.
    *   This doesn't currently implement support for null values, but could.
    *   Uses numba not cython
    """

    assert axis in (0, 1)
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else: error("sparse mean variance only works on sparse csr and csc matrices")

    if axis == ax_minor:
        return sparse_mean_var_major_axis(
            mtx.data,
            mtx.indptr,
            major_len = shape[0],
            minor_len = shape[1],
            n_threads = numba.get_num_threads(),
        )
    else:
        return sparse_mean_var_minor_axis(
            mtx.data,
            mtx.indices,
            mtx.indptr,
            major_len = shape[0],
            minor_len = shape[1],
            n_threads = numba.get_num_threads(),
        )


@njit
def sparse_mean_var_minor_axis(
    data, indices, indptr, *, major_len, minor_len, n_threads
):
    """
    Compute mean and variance for a sparse matrix for the minor axis.
    Given arrays for a csr matrix, returns the means and variances for each column back.
    """

    rows = len(indptr) - 1
    sums_minor = np.zeros((n_threads, minor_len))
    squared_sums_minor = np.zeros((n_threads, minor_len))
    means = np.zeros(minor_len)
    variances = np.zeros(minor_len)
    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            for j in range(indptr[r], indptr[r + 1]):
                minor_index = indices[j]
                if minor_index >= minor_len:
                    continue
                value = data[j]
                sums_minor[i, minor_index] += value
                squared_sums_minor[i, minor_index] += value * value 

    for c in numba.prange(minor_len):
        sum_minor = sums_minor[:, c].sum()
        means[c] = sum_minor / major_len
        variances[c] = (
            squared_sums_minor[:, c].sum() / major_len - (sum_minor / major_len) ** 2
        )

    return means, variances


@njit
def sparse_mean_var_major_axis(data, indptr, *, major_len, minor_len, n_threads):
    """
    Compute mean and variance for a sparse array for the major axis.
    Given arrays for a csr matrix, returns the means and variances for each row back.
    """

    rows = len(indptr) - 1
    means = np.zeros(major_len)
    variances = np.zeros_like(means)

    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            sum_major = 0.0
            squared_sum_minor = 0.0
            for j in range(indptr[r], indptr[r + 1]):
                value = np.float64(data[j])
                sum_major += value
                squared_sum_minor += value * value
            means[r] = sum_major
            variances[r] = squared_sum_minor

    for c in numba.prange(major_len):
        mean = means[c] / minor_len
        means[c] = mean
        variances[c] = variances[c] / minor_len - mean * mean

    return means, variances


@numba.njit(cache = True, parallel = False)
def sum_and_sum_squares_clipped(
    indices, data, *,
    n_cols: int, clip_val, nnz
):
    squared_batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    for i in numba.prange(nnz):
        idx = indices[i]
        element = min(np.float64(data[i]), clip_val[idx])
        squared_batch_counts_sum[idx] += element**2
        batch_counts_sum[idx] += element
    return squared_batch_counts_sum, batch_counts_sum


def highly_variable_genes_seurat(
    E, *, obs, var,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    sort_hvg_rank_first = False,
    span: float = 0.3,
    subset: bool = False
) -> pd.DataFrame | None:
    """
    Highly variable gene filter done by seurat v3 and above. This method is based on
    regularized standard deviation after variable stabilizing transformation, as similarly
    described for the sctransform methods. This requires input matrix be raw counts E.

    Returns
    ---------
    
    highly_variable :
        Boolean indicator of highly-variable genes.

    means
        Means per gene.

    variances
        Variance per gene.

    variances_norm
        Normalized variance per gene, averaged in the case of multiple batches.

    highly_variable_rank
        Rank of the gene according to normalized variance, median rank in the 
        case of multiple batches.
    
    highly_variable_nbatches
        If ``batch_key`` is given, this denotes in how many batches genes are 
        detected as highly variable genes.
    """

    try: from skmisc.loess import loess
    except ImportError as e:
        error('requires the installation of package `scikit-misc`.')
    
    df = pd.DataFrame(index = var.index)
    data = E

    df["means"], df["variances"] = mean_variance(data)

    if batch_key is None: batch_info = pd.Categorical(np.zeros(E.shape[0], dtype = int))
    else: batch_info = obs[batch_key].to_numpy()

    norm_gene_vars = []
    for b in np.unique(batch_info):
        data_batch = data[batch_info == b]

        mean, var = mean_variance(data_batch)
        not_const = var > 0
        estimat_var = np.zeros(data.shape[1], dtype=np.float64)

        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        model = loess(x, y, span = span, degree = 2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10**estimat_var)

        # clip large values as in Seurat
        N = data_batch.shape[0]
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + mean
        if isinstance(data_batch, sparse.csr_matrix) or isinstance(data_batch, sparse.csc_matrix):
            batch_counts = data_batch.tocsr()
            squared_batch_counts_sum, batch_counts_sum = sum_and_sum_squares_clipped(
                batch_counts.indices,
                batch_counts.data,
                n_cols=batch_counts.shape[1],
                clip_val=clip_val,
                nnz=batch_counts.nnz,
            )

        else:
            batch_counts = data_batch.astype(np.float64).copy()
            clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
            np.putmask(
                batch_counts,
                batch_counts > clip_val_broad,
                clip_val_broad,
            )

            squared_batch_counts_sum = np.square(batch_counts).sum(axis = 0)
            batch_counts_sum = batch_counts.sum(axis = 0)

        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis = 0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis = 1), axis = 1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis = 0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df["gene"] = df.index
    df["hvg.n"] = num_batches_high_var
    df["hvg.rank"] = median_ranked
    df["var.norm"] = np.mean(norm_gene_vars, axis = 0)
    
    if sort_hvg_rank_first:
        sort_cols = ["hvg.rank", "hvg.n"]
        sort_ascending = [True, False]
    else:
        sort_cols = ["hvg.n", "hvg.rank"]
        sort_ascending = [False, True]
    
    sorted_index = (
        df[sort_cols]
        .sort_values(sort_cols, ascending = sort_ascending, na_position = "last")
        .index
    )

    df["hvg"] = False
    df.loc[sorted_index[: int(n_top_genes)], "hvg"] = True
    if batch_key is None: df = df.drop(["hvg.n"], axis = 1)
    if subset: df = df.iloc[df["hvg"].to_numpy(), :]
    return df


def materialize_as_ndarray(a):
    return tuple(np.asarray(arr) for arr in a)


def highly_variable_nb(
    X, var, *, n_bins: int = 20,
) -> pd.DataFrame:
    '''
    Highly variable gene filter in legacy version of seurat. 
    This method is based on binned dispersion method. 
    This requires input matrix be raw counts X, as dispersions are fitted on UMI.
    '''

    mean, var = materialize_as_ndarray(mean_variance(X))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)

    # all of the following quantities are "per-gene" here
    df = pd.DataFrame(
        dict(zip(["means", "dispersions"], (mean, dispersion), strict = True))
    )

    df["mean.bin"] = mean_bins(df["means"], flavor = 'seurat', n_bins = n_bins)
    disp_stat = disp_stats(df, flavor = 'seurat')

    # actually do the normalization
    df["norm.disp"] = (df["dispersions"] - disp_stat["avg"]) / disp_stat["dev"]
    df.index = var.index
    return df


def highly_variable_nb_log(
    X, var, *, n_bins: int = 20,
) -> pd.DataFrame:
    '''
    Highly variable gene filter in legacy version of seurat. 
    This method is based on binned dispersion method. 
    This requires input matrix be log normalized counts.
    '''

    mean, var = materialize_as_ndarray(mean_variance(X))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    
    # all of the following quantities are "per-gene" here
    df = pd.DataFrame(
        dict(zip(["means", "dispersions"], (mean, dispersion), strict = True))
    )

    df["mean.bin"] = mean_bins(df["means"], flavor = 'cellranger', n_bins = n_bins)
    disp_stat = disp_stats(df, flavor = 'cellranger')

    # actually do the normalization
    df["norm.disp"] = (df["dispersions"] - disp_stat["avg"]) / disp_stat["dev"]
    df.index = var.index
    return df


def mean_bins(means: pd.Series, flavor, n_bins: int) -> pd.Series:

    if flavor == "seurat": bins = n_bins
    elif flavor == "cellranger":
        bins = np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf]
    else: error('invalid flavor.')
    return pd.cut(means, bins = bins)


def disp_stats(df: pd.DataFrame, flavor) -> pd.DataFrame:

    disp_grouped = df.groupby("mean.bin", observed = True)["dispersions"]
    if flavor == "seurat":
        disp_bin_stats = disp_grouped.agg(avg = "mean", dev = "std")
        mean_bin = df["mean.bin"]

        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1

        one_gene_per_bin = disp_bin_stats["dev"].isnull()
        gen_indices = np.flatnonzero(one_gene_per_bin.loc[mean_bin])
        if len(gen_indices) == 0: return
        disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[one_gene_per_bin, "avg"]
        disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0

    elif flavor == "cell_ranger":

        def mean_abs_dev(a):
            from statsmodels.robust import mad
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category = RuntimeWarning)
                return mad(a)
            
        disp_bin_stats = disp_grouped.agg(avg = "median", dev = mean_abs_dev)
    
    else: error('invalid flavor.')
    return disp_bin_stats.loc[df["mean.bin"]].set_index(df.index)


def stable_genes(E, modality = 'rna', stable_number = None, use_signal_to_noise = True):
    '''
    Select stable features for modalities. Each modalities is characterized with an expected amount
    of stable features that are capable of fitting the multiplet distribution function.
    This function is only revelant in ``composite.py`` (the ``sccomposite`` routine).

    Parameters
    -----------

    E : np.array | scipy.sparse.csc_matrix | scipy.sparse.csr_matrix
        numpy array of feature counts matrix, either dense or sparse.
    
    modality : Literal['rna' | 'atac' | 'adt']
        Which modality do the matrix come from. This is used to specify expected stable feature
        counts (300 at least for scRNA dataset and scATAC datasets, and 15 for ADT.)
    
    stable_number : int | None
        Manually specify the number of expected stable features. This will override the modality
        parameter used for deciding them automatically.

    use_signal_to_noise : bool
        Use signal to noise ratio to determine selection.
    
    Returns
    ---------
    Returns the top ``stable_number`` signal-to-noise ratio genes. The returning matrix has a
    shape of (n_cells, n_genes).
    '''
    
    full = E.T
    if sparse.issparse(full) == True:
        full = full.todense()

    if stable_number == None:
        if modality == 'rna':
            stable_number = 300
        elif modality == 'atac':
            stable_number = 300
        elif modality == 'adt':
            stable_number = 15

    nonzero_proportion = np.array((full > 0).sum(axis = 1)) / full.shape[1]
    full = np.array(full)
    subset = full[np.squeeze(nonzero_proportion > 0.5),]
    high_nonzero = full[np.squeeze(nonzero_proportion > 0.5),]
    high_nonzero = np.log1p(high_nonzero)

    if high_nonzero.shape[0] < stable_number:
        warning('too few stable features to provide reliable inference.')
        result = subset

    elif use_signal_to_noise:
        mean = np.mean(high_nonzero, axis = 1)
        std = np.std(high_nonzero, axis = 1)
        signal = mean / std # signal to noise ratio

        # calculate the rank of the magnitude of signal for each cell
        order = (- signal).argsort()
        ranks = order.argsort()
        result = subset[ranks < stable_number]

    else:
        mean = np.mean(high_nonzero, axis = 1)
        order = (- mean).argsort()
        ranks = order.argsort()
        result = subset[ranks < stable_number]

    result = np.transpose(result)
    return result
