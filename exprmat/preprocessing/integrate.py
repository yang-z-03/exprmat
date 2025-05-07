
import anndata as ad
import numpy as np
from exprmat.ansi import error, warning


def bbknn(  # noqa: PLR0913
    adata: ad.AnnData,
    *,
    key: str = "batch",
    use_rep: str = "pca",
    approx: bool = True,
    use_annoy: bool = True,
    metric: str = "euclidean",
    copy: bool = False,
    neighbors_within_batch: int = 3,
    n_pcs: int = 50,
    trim: int | None = None,
    annoy_n_trees: int = 10,
    pynndescent_n_neighbors: int = 30,
    pynndescent_random_state: int = 0,
    use_faiss: bool = True,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    **kwargs,
) -> ad.AnnData | None:
    """
    Batch balanced kNN

    Batch balanced kNN alters the kNN procedure to identify each cell's top neighbours in
    each batch separately instead of the entire cell pool with no accounting for batch.
    The nearest neighbours for each batch are then merged to create a final list of
    neighbours for the cell. Aligns batches in a quick and lightweight manner.
    This batch correction method is used as a drop-in replacement of nearest neighbor
    search instead of kNN. It should be run on 

    Parameters
    -----------
    adata
        Needs the PCA computed and stored in `adata.obsm["pca"]`.
    
    key
        `adata.obs` column name discriminating between your batches.
    
    use_rep
        The dimensionality reduction in `.obsm` to use for neighbour detection. Defaults to PCA.
    
    approx
        If `True`, use approximate neighbour finding - annoy or PyNNDescent. This results in a 
        quicker run time for large datasets while also potentially increasing the degree of batch correction.
    
    use_annoy
        Only used when `approx = True`. If `True`, will use annoy for neighbour finding. If
        `False`, will use pynndescent instead.

    metric
        What distance metric to use. The options depend on the choice of neighbour algorithm.
        "euclidean", the default, is always available. Annoy supports "angular", "manhattan" 
        and "hamming". PyNNDescent supports metrics listed in `pynndescent.distances.named_distances`
        and custom functions, including compiled Numba code. ['euclidean', 'l2', 'sqeuclidean', 
        'manhattan', 'taxicab', 'l1', 'chebyshev', 'linfinity', 'linfty', 'linf', 'minkowski', 
        'seuclidean', 'standardised_euclidean', 'wminkowski', ...])
        
    neighbors_within_batch
        How many top neighbours to report for each batch; total number of neighbours in
        the initial k-nearest-neighbours computation will be this number times the number
        of batches. This then serves as the basis for the construction of a symmetrical
        matrix of connectivities.

    n_pcs
        How many dimensions (in case of PCA, principal components) to use in the analysis.

    trim
        Trim the neighbours of each cell to these many top connectivities. May help with
        population independence and improve the tidiness of clustering. The lower the value the
        more independent the individual populations, at the cost of more conserved batch effect.
        If `None`, sets the parameter value automatically to 10 times `neighbors_within_batch`
        times the number of batches. Set to 0 to skip.

    annoy_n_trees
        Only used with annoy neighbour identification. The number of trees to construct in the
        annoy forest. More trees give higher precision when querying, at the cost of increased
        run time and resource intensity.

    pynndescent_n_neighbors
        Only used with pyNNDescent neighbour identification. The number of neighbours to include
        in the approximate neighbour graph. More neighbours give higher precision when querying,
        at the cost of increased run time and resource intensity.

    pynndescent_random_state
        Only used with pyNNDescent neighbour identification. The RNG seed to use when creating the graph.

    use_faiss
        If `approx = False` and the metric is "euclidean", use the faiss package to compute
        nearest neighbours if installed. This improves performance at a minor cost to numerical
        precision as faiss operates on float32.

    set_op_mix_ratio
        UMAP connectivity computation parameter, float between 0 and 1, controlling the
        blend between a connectivity matrix formed exclusively from mutual nearest neighbour
        pairs (0) and a union of all observed neighbour relationships with the mutual pairs
        emphasised (1)

    local_connectivity
        UMAP connectivity computation parameter, how many nearest neighbors of each cell
        are assumed to be fully connected (and given a connectivity value of 1)
    """

    try: from bbknn import bbknn
    except ImportError as e: error("bbknn required installation of package `bbknn`.")
    
    return bbknn(
        adata = adata,
        batch_key = key,
        use_rep = use_rep,
        approx = approx,
        use_annoy = use_annoy,
        metric = metric,
        copy = copy,
        neighbors_within_batch = neighbors_within_batch,
        n_pcs = n_pcs,
        trim = trim,
        annoy_n_trees = annoy_n_trees,
        pynndescent_n_neighbors = pynndescent_n_neighbors,
        pynndescent_random_state = pynndescent_random_state,
        use_faiss = use_faiss,
        set_op_mix_ratio = set_op_mix_ratio,
        local_connectivity = local_connectivity,
        **kwargs,
    )


def harmony(
    adata: ad.AnnData,
    key: str,
    *,
    basis: str = "pca",
    adjusted_basis: str = "harmony",
    **kwargs,
):
    """
    Use harmonypy to integrate different experiments.

    Harmony is an algorithm for integrating single-cell data from multiple experiments. 
    This function uses the python port of Harmony, ``harmonypy``, to integrate single-cell data
    stored in an AnnData object. As Harmony works by adjusting the principal components, 
    this function should be run after performing PCA but before computing the neighbor graph.

    Parameters
    ----------
    adata
        The annotated data matrix.

    key
        The name of the column in ``adata.obs`` that differentiates among experiments / batches. 
        To integrate over two or more covariates, you can pass multiple column names as a list. 
        See ``vars_use`` parameter of the ``harmonypy`` package for more details.
    
    basis
        The name of the field in ``adata.obsm`` where the PCA table is stored. Defaults to 'pca'.
    
    adjusted_basis
        The name of the field in ``adata.obsm`` where the adjusted PCA table.

    kwargs
        theta = None,
        lamb = None,
        sigma = 0.1, 
        nclust = None,
        tau = 0,
        block_size = 0.05, 
        max_iter_harmony = 10,
        max_iter_kmeans = 20,
        epsilon_cluster = 1e-5,
        epsilon_harmony = 1e-4, 
        plot_convergence = False,
        verbose = True,
        reference_values = None,
        cluster_prior = None,
        random_state = 0,
        cluster_fn = 'kmeans'
    """

    try: import harmonypy
    except ImportError as e: 
        error('harmony integration requires installation of package `harmonypy`.')

    X = adata.obsm[basis].astype(np.float64)
    default_params = {
        'theta': None,
        'lamb': None,
        'sigma': 0.1, 
        'nclust': None,
        'tau': 0,
        'block_size': 0.05, 
        'max_iter_harmony': 10,
        'max_iter_kmeans': 20,
        'epsilon_cluster': 1e-5,
        'epsilon_harmony': 1e-4, 
        'plot_convergence': False,
        'verbose': False,
        'reference_values': None,
        'cluster_prior': None,
        'random_state': 0,
        'cluster_fn': 'kmeans'
    }

    default_params.update(kwargs)
    harmony_out = harmonypy.run_harmony(X, adata.obs, key, **default_params)
    adata.obsm[adjusted_basis] = harmony_out.Z_corr.T
    return adata


def scanorama(
    adata: ad.AnnData,
    key: str,
    *,
    basis: str = "pca",
    adjusted_basis: str = "scanorama",
    knn: int = 20,
    sigma: float = 15,
    approx: bool = True,
    alpha: float = 0.10,
    batch_size: int = 5000,
    ** kwargs,
) -> None:
    """
    Use Scanorama to integrate different experiments.

    Scanorama is an algorithm for integrating single-cell data from multiple experiments 
    stored in an AnnData object. This function should be run after performing PCA but before 
    computing the neighbor graph, as illustrated in the example below.

    Parameters
    ----------
    adata
        The annotated data matrix.

    key
        The name of the column in ``adata.obs`` that differentiates among experiments / batches. 
        Cells from the same batch must be contiguously stored in ``adata``.

    basis
        The name of the field in ``adata.obsm`` where the PCA table is stored.

    adjusted_basis
        The name of the field in ``adata.obsm`` where the integrated embeddings will be stored 
        after running this function. Defaults to ``scanorama``.
    
    knn
        Number of nearest neighbors to use for matching.
    
    sigma
        Correction smoothing parameter on Gaussian kernel.
    
    approx
        Use approximate nearest neighbors with Python ``annoy`` greatly speeds up matching runtime.
    
    alpha
        Alignment score minimum cutoff.
    
    batch_size
        The batch size used in the alignment vector computation. Useful when integrating very 
        large (> 100k samples) datasets. Set to large value that runs within available memory.

    kwargs
        Any additional arguments will be passed to ``scanorama.assemble()``.
    """

    try: import scanorama
    except ImportError as e:
        error('scanorama integration requires installation of package `scanorama`.')

    # get batch indices in linear time.
    curr_batch = None
    batch_names = []
    name2idx = {}
    
    for idx in range(adata.X.shape[0]):
        batch_name = adata.obs[key].iat[idx]
        if batch_name != curr_batch:
            curr_batch = batch_name
            if batch_name in batch_names:
                # contiguous batches important for preserving cell order.
                error("detected non-contiguous batches.")
            batch_names.append(batch_name)  # Preserve name order.
            name2idx[batch_name] = []
        name2idx[batch_name].append(idx)

    # separate batches.
    datasets_dimred = [
        adata.obsm[basis][name2idx[batch_name]] 
        for batch_name in batch_names
    ]

    # integrate.
    integrated = scanorama.assemble(
        datasets_dimred,  # assemble in low dimensional space.
        knn = knn,
        sigma = sigma,
        approx = approx,
        alpha = alpha,
        ds_names = batch_names,
        batch_size = batch_size,
        **kwargs
    )

    adata.obsm[adjusted_basis] = np.concatenate(integrated)
    return adata