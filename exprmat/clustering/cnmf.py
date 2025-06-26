
import numpy as np
import pandas as pd
import os, errno, sys
import datetime
import uuid
import itertools
import yaml
import subprocess
import scipy.sparse as sp
import warnings

from scipy.spatial.distance import squareform
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import sparsefuncs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import leaves_list, linkage
from multiprocessing import Pool

from exprmat.dynamics.generank import get_mean_var
from exprmat.utils import choose_layer
from exprmat.ansi import warning, error, info


def worker_filter(iterable, worker_index, total_workers):
    return (p for i, p in enumerate(iterable) if (i - worker_index) % total_workers == 0)


def ols(X, Y, batch_size = 1024, normalize_y = False):
    """
    Solve ordinary least squares: b = (X^T X)^-1 X^T Y.
    and accumulate X^T X and X^T Y in row-batches.
    
    Optionally mean/variance-normalize each column of Y globally (using the entire 
    dataset's mean/var), while still only converting each row-batch to dense on-the-fly.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_predictors)
        Predictor matrix.

    Y : np.ndarray or scipy.sparse.spmatrix, shape (n_samples, n_targets)
        Outcomes. Each column is one target variable.

    batch_size : int
        Number of rows to process per chunk.

    normalize_y : bool
        If True, compute global mean & var of Y columns, then subtract mean 
        and divide by std for each batch.

    Returns
    -------
    Beta : np.ndarray, shape (n_predictors, n_targets)
        The OLS coefficients for each target.
    """

    n_samples, n_predictors = X.shape
    n_samples_Y, n_targets = Y.shape
    if n_samples != n_samples_Y:
        error("x and y must have the same number of rows.")

    # optionally compute global mean & variance of Y columns
    if normalize_y:
        meanY, varY = get_mean_var(Y)

        # avoid zero or near-zero std
        eps = 1e-12
        varY[varY < eps] = eps
        stdY = np.sqrt(varY)

    # initialize accumulators
    XtX = np.zeros((n_predictors, n_predictors), dtype = np.float64)
    XtY = np.zeros((n_predictors, n_targets),    dtype = np.float64)

    # process rows in batches
    for start_row in range(0, n_samples, batch_size):
        end_row = min(start_row + batch_size, n_samples)
        X_batch = X[start_row:end_row, :]

        # extract chunk from Y.  if sparse, convert only this subset to dense.
        if sp.issparse(Y) and normalize_y:
            # only need to densify if normalizing
            Y_batch = Y[start_row:end_row, :].toarray()
        else: Y_batch = Y[start_row:end_row, :]

        # optionally apply normalization
        if normalize_y: Y_batch = (Y_batch - meanY) / stdY
        
        # accumulate partial sums
        XtX += X_batch.T @ X_batch
        XtY += X_batch.T @ Y_batch

    beta, residuals, rank, s = np.linalg.lstsq(XtX, XtY, rcond=None)
    return beta


def factorize_mp_signature(args):
    """
    factorize_mp_signature(
        counts: Any,
        replicate_kwargs: Any,
        run_kwargs: Any,
        worker_i: int = 0,
        total_workers: int = 1
    )
    """
    return factorize(*args)


def prepare_cnmf(
    adata, components, counts = 'counts', tpm = 'norm', hvg = 'hvg.vst', min_counts = 3,
    alpha_usage = 0.0, alpha_spectra = 0.0, init = 'random', max_nmf_iter = 1000,
    n_iter = 100, seed = 42, beta_loss = 'frobenius'
):

    import anndata as ad

    # prepare the subset of highly variable genes only.
    c = choose_layer(adata, layer = counts)
    c = ad.AnnData(X = c, obs = adata.obs, var = adata.var)
    c.layers['norm'] = adata.layers[tpm]
    subset = c[:, c.var[hvg]].copy()

    if sp.issparse(subset.layers['norm']):
        gene_tpm_mean, gene_tpm_stddev = get_mean_var(subset.layers['norm'])
        gene_tpm_stddev = gene_tpm_stddev ** .5
    else:
        gene_tpm_mean = np.array(subset.layers['norm'].mean(axis = 0)).reshape(-1)
        gene_tpm_stddev = np.array(subset.layers['norm'].std(axis = 0, ddof = 0)).reshape(-1)
    
    subset.var['hvg.mean'] = gene_tpm_mean
    subset.var['hvg.stdev'] = gene_tpm_stddev

    # filter cells
    keepcells = np.array(subset.X.sum(axis = 1) >= min_counts).reshape(-1)
    subset = subset[keepcells, :].copy()

    (replicate_params, run_params) = get_nmf_iter_params(
        ks = components, n_iter = n_iter, random_state_seed = seed,
        beta_loss = beta_loss, alpha_usage = alpha_usage,
        alpha_spectra = alpha_spectra, init = init, max_iter = max_nmf_iter
    )
    
    return subset, replicate_params, run_params


def get_nmf_iter_params(
    ks, n_iter = 100, random_state_seed = 42,
    beta_loss = 'kullback-leibler',
    alpha_usage = 0.0, alpha_spectra = 0.0,
    init = 'random', max_iter = 1000):
    """
    Create a DataFrame with parameters for NMF iterations.


    Parameters
    ----------
    ks : integer, or list-like.
        Number of topics (components) for factorization.
        Several values can be specified at the same time, which will be run independently.

    n_iter : integer, optional (defailt=100)
        Number of iterations for factorization. If several ``k`` are specified, this many
        iterations will be run for each value of ``k``.

    random_state_seed : int or None, optional (default=None)
        Seed for sklearn random state.
        
    alpha_usage : float, optional (default=0.0)
        Regularization parameter for NMF corresponding to alpha_W in scikit-learn

    alpha_spectra : float, optional (default=0.0)
        Regularization parameter for NMF corresponding to alpha_H in scikit-learn
    """

    if type(ks) is int: ks = [ks]

    # remove any repeated k values, and order.
    k_list = sorted(set(list(ks)))

    n_runs = len(ks) * n_iter
    np.random.seed(seed = random_state_seed)
    nmf_seeds = np.random.randint(low = 1, high = (2**31)-1, size = n_runs)

    replicate_params = []
    for i, (k, r) in enumerate(itertools.product(k_list, range(n_iter))):
        replicate_params.append([k, r, nmf_seeds[i]])
    replicate_params = pd.DataFrame(replicate_params, columns = ['n_comps', 'iter', 'nmf_seed'])
    
    nmf_kwargs = dict(
        alpha_w = alpha_usage,
        alpha_h = alpha_spectra,
        l1_ratio = 0.0,
        beta_loss = beta_loss,
        solver = 'mu',
        tolerance = 1e-4,
        max_iter = max_iter,
        init = init
    )
    
    # coordinate descent is faster than multiplicative update but only works for frobenius
    if beta_loss == 'frobenius': nmf_kwargs['solver'] = 'cd'
    return(replicate_params, nmf_kwargs)


def factorize(
    counts, replicate_kwargs, run_kwargs,
    worker_i = 0, total_workers = 1
):
    """
    Iteratively run NMF with prespecified parameters.
    Use the `worker_i` and `total_workers` parameters for parallelization.
    """

    jobs_for_this_worker = worker_filter(range(len(replicate_kwargs)), worker_i, total_workers)

    results = {}
    for idx in jobs_for_this_worker:
        p = replicate_kwargs.iloc[idx, :]
        run_kwargs['random_state'] = p['nmf_seed']
        run_kwargs['n_comps'] = p['n_comps']

        from exprmat.clustering.nmf import nmf
        (usages, spectra, nit) = nmf(counts.X, **run_kwargs)
        spectra = pd.DataFrame(
            spectra,
            index = np.arange(1, run_kwargs['n_comps'] + 1),
            columns = counts.var.index
        )
        
        results[f'{run_kwargs["n_comps"]}.{p["iter"]}'] = spectra
    
    return results


def combine_nmf(counts, k, replicate_kwargs, run_kwargs, workers = 1):

    subset_kwargs = replicate_kwargs[replicate_kwargs.n_comps == k].sort_values('iter')
    combined_spectra = []
    dicts = []

    list_args = [(counts, subset_kwargs, run_kwargs, x, workers) for x in range(workers)]
    with Pool(workers) as p:
        dicts = p.map(factorize_mp_signature, list_args)
        p.close()
        p.join()
    
    for d in dicts:
        for key in d.keys():
            combined_spectra.append(d[key])

    if len(combined_spectra) > 0:        
        combined_spectra = pd.concat(combined_spectra, axis = 0)
    return combined_spectra


def combine(counts, replicate_kwargs, run_kwargs, components = None, workers = 1):
    
    if type(components) is int:
        ks = [components]
    elif components is None:
        ks = sorted(set(replicate_kwargs.n_comps))
    else: ks = components

    combined = {}
    for k in ks: 
        combined[k] = combine_nmf(counts, k, replicate_kwargs, run_kwargs, workers)
    return combined


def refit_usage(X, spectra, run_kwargs):
    """
    Takes an input data matrix and a fixed spectra and uses NNLS to find the optimal
    usage matrix. If input data are pandas dataframes, returns a dataframe with row 
    index matching X and columns index matching index of spectra.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray, cells * genes
        Non-negative expression data to fit spectra to

    spectra : pandas.DataFrame or numpy.ndarray, programs * genes
        Non-negative spectra of expression programs
    """

    refit_nmf_kwargs = run_kwargs

    if type(spectra) is pd.DataFrame:
        refit_nmf_kwargs.update(dict(
            n_comps = spectra.shape[0], 
            H = spectra.values.astype(X.dtype), 
            update_h = False
        ))

    else:
        refit_nmf_kwargs.update(dict(
            n_comps = spectra.shape[0], 
            H = spectra.astype(X.dtype), 
            update_h = False
        ))
        
    from exprmat.clustering.nmf import nmf
    rf_usages, _, _ = nmf(X, **refit_nmf_kwargs)
    if (type(X) is pd.DataFrame) and (type(spectra) is pd.DataFrame):
        rf_usages = pd.DataFrame(rf_usages, index = X.index, columns = spectra.index)
    return rf_usages


def refit_spectra(X, usage, run_kwargs):
    return refit_usage(X.T, usage.T, run_kwargs).T


def consensus(
    adata, combined_spectras, k, density_threshold = 0.5, local_neighborhood_size = 0.30,
    run_kwargs = None,
    refit = True, normalize_tpm_spectra = False, random_state = 42, stat_only = False,
):
    merged_spectra = combined_spectras[k]
    norm_counts = adata.X
    density_threshold_str = str(density_threshold)
    n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0] / k)

    # rescale topics such to length of 1.
    l2_spectra = (merged_spectra.T / np.sqrt((merged_spectra**2).sum(axis = 1))).T

    if not stat_only:

        # compute the local density matrix (if not previously cached)
        # first find the full distance matrix
        topics_dist = euclidean_distances(l2_spectra.values)
        # partition based on the first n neighbors
        partitioning_order  = np.argpartition(topics_dist, n_neighbors + 1)[:, :n_neighbors + 1]
        # find the mean over those n_neighbors (excluding self, which has a distance of 0)
        distance_to_nearest_neighbors = topics_dist[
            np.arange(topics_dist.shape[0])[:, None], 
            partitioning_order
        ]
        
        local_density = pd.DataFrame(
            distance_to_nearest_neighbors.sum(1) / (n_neighbors),
            columns = ['local.density'],
            index = l2_spectra.index
        )
        
        del partitioning_order
        del distance_to_nearest_neighbors

        density_filter = local_density.iloc[:, 0] < density_threshold
        l2_spectra = l2_spectra.loc[density_filter, :]
        if l2_spectra.shape[0] == 0:
            warning("consider increasing density threshold.")
            error("zero components remain after density filtering.")

        topics_dist = topics_dist[density_filter.values, :][:, density_filter.values]

    kmeans_model = KMeans(n_clusters = k, n_init = 10, random_state = random_state)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_ + 1, index = l2_spectra.index)

    # find median usage for each gene across cluster
    median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

    # normalize median spectra to probability distributions.
    median_spectra = (median_spectra.T/median_spectra.sum(1)).T

    # obtain reconstructed count matrix by re-fitting usage and computing 
    # dot product: usage.dot(spectra)
    rf_usages = refit_usage(norm_counts, median_spectra, run_kwargs)
    rf_usages = pd.DataFrame(rf_usages, index = adata.obs.index, columns = median_spectra.index)        
    
    silhouette = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric = 'euclidean')
    
    # compute prediction error as a frobenius norm
    rf_pred_norm_counts = rf_usages.dot(median_spectra)        
    if sp.issparse(norm_counts):
        prediction_error = ((norm_counts.todense() - rf_pred_norm_counts)**2).sum().sum()
    else: prediction_error = ((norm_counts - rf_pred_norm_counts)**2).sum().sum()    
        
    # ('k', 'local_density_threshold', 'silhouette', 'prediction_error')
    consensus_stats = (k, density_threshold, silhouette,  prediction_error)     
    
    if stat_only: return consensus_stats   
    
    # re-order usage by total contribution
    norm_usages = rf_usages.div(rf_usages.sum(axis = 1), axis = 0)      
    reorder = norm_usages.sum(axis =0).sort_values(ascending = False)
    rf_usages = rf_usages.loc[:, reorder.index]
    norm_usages = norm_usages.loc[:, reorder.index]
    median_spectra = median_spectra.loc[reorder.index, :]
    rf_usages.columns = np.arange(1, rf_usages.shape[1] + 1)
    norm_usages.columns = rf_usages.columns
    median_spectra.index = rf_usages.columns
    
    # convert spectra to tpm units, and obtain results for all genes by running 
    # last step of nmf with usages fixed and tpm as the input matrix.

    spectra_tpm = refit_spectra(
        adata.layers['norm'], 
        norm_usages.astype(adata.layers['norm'].dtype),
        run_kwargs = run_kwargs
    )
    
    spectra_tpm = pd.DataFrame(spectra_tpm, index = rf_usages.columns, columns = adata.var.index)
    if normalize_tpm_spectra: spectra_tpm = spectra_tpm.div(spectra_tpm.sum(axis = 1), axis = 0) * 1e6
    
    # convert spectra to z-score units by fitting old regression of the 
    # z-scored tpm against gep usage
    usage_coef = ols(rf_usages.values, adata.layers['norm'], normalize_y = True)
    usage_coef = pd.DataFrame(usage_coef, index = rf_usages.columns, columns = adata.var.index)
    
    if refit:

        import scanpy as sc
        if sp.issparse(adata.layers['norm']):
            scaled_tpm = sc.pp.scale(adata.layers['norm'], zero_center = False, copy = True)                       
        else: scaled_tpm = adata.layers['norm'] / adata.layers['norm'].std(axis = 0, ddof = 1)
            
        spectra_tpm_rf = spectra_tpm
        spectra_tpm_rf = spectra_tpm_rf.div(adata.var['hvg.stdev'], axis = 1)
        rf_usages = refit_usage(
            scaled_tpm, spectra_tpm_rf.astype(scaled_tpm.dtype), 
            run_kwargs = run_kwargs
        )
        
        rf_usages = pd.DataFrame(rf_usages, index = adata.obs.index, columns = spectra_tpm_rf.index)                                                                  
    

    spectra_order = []

    for cl in sorted(set(kmeans_cluster_labels)):
        cl_filter = (kmeans_cluster_labels == cl)

        if cl_filter.sum() > 1:
            cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter], checks=False)
            cl_dist[cl_dist < 0] = 0 # rarely get floating point arithmetic issues
            cl_link = linkage(cl_dist, 'average')
            cl_leaves_order = leaves_list(cl_link)
            spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
        else: spectra_order += list(np.where(cl_filter)[0])

    dist = topics_dist[spectra_order, :][:, spectra_order]
    return local_density, dist, median_spectra, rf_usages, spectra_tpm, usage_coef, consensus_stats


def cnmf(
    adata, ks, comb = None, counts: str = 'counts', tpm: str = 'norm', hvg: str = 'hvg.vst', 
    min_counts: int = 3, 
    alpha_usage: float = 0, alpha_spectra: float = 0, 
    init: str = 'random', 
    max_nmf_iter: int = 1000, n_iter: int = 100, 
    seed: int = 42, beta_loss: str = 'frobenius', 
    ncpus = 1, density_threshold: float = 0.5,
    local_neighborhood_size: float = 0.3,
    refit: bool = True,
    normalize_tpm_spectra: bool = False, return_k = None, 
):

    prep, replicate_kw, run_kw = prepare_cnmf(
        adata, components = ks,
        counts = counts, tpm = tpm, hvg = hvg, min_counts = min_counts,
        alpha_usage = alpha_usage, alpha_spectra = alpha_spectra,
        init = init, max_nmf_iter = max_nmf_iter, n_iter = n_iter,
        seed = seed, beta_loss = beta_loss
    )

    if comb is None:
        comb = combine(
            prep, replicate_kwargs = replicate_kw, run_kwargs = run_kw,
            components = ks, workers = ncpus
        )

    if return_k is None:
        stats = {}
        for k in ks:
            stats[k] = consensus(
                prep, comb, run_kwargs = run_kw, k = k, 
                stat_only = True, density_threshold = density_threshold,
                local_neighborhood_size = local_neighborhood_size,
                refit = refit, normalize_tpm_spectra = normalize_tpm_spectra
            )
        return comb, stats
    
    res = consensus(
        prep, comb, run_kwargs = run_kw, k = return_k, 
        stat_only = False, density_threshold = density_threshold,
        local_neighborhood_size = local_neighborhood_size,
        refit = refit, normalize_tpm_spectra = normalize_tpm_spectra
    )

    return comb, res

