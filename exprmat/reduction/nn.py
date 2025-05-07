
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.sparse

from exprmat.ansi import warning, error, info
from exprmat.utils import choose_representation, is_constant


def knn_graph(
    embedding, k = 5, dist_metric = 'euclidean', approx = False, 
    random_seed = 42
):
    '''
    Calculate exact or approximate kNN graph.
    Using approximate kNN requires installation of package ``annoy``.

    Returns
    ----------

    A tuple of three:
    
    *   edges of the graph (n_edges, 2)
    *   nearest neighbor index matrix (n_cells, n_neighbors)
    *   distance matrix (n_cells, n_cells)
    '''

    if approx:
        try: from annoy import AnnoyIndex
        except:
            approx = False
            warning(f'approximate nn graph requires installation of `annoy`.')
            warning(f'fall back to brute force neighbors retrieval.')
    
    if approx:
        
        if dist_metric == 'cosine': dist_metric = 'angular'
        npc = embedding.shape[1]
        ncell = embedding.shape[0]
        annoy_index = AnnoyIndex(npc, metric = dist_metric)
        annoy_index.set_seed(random_seed)

        for i in range(ncell):
            annoy_index.add_item(i, list(embedding[i,:]))
        annoy_index.build(10) # 10 trees

        knn = []
        for cell_id in range(ncell):
            knn.append(annoy_index.get_nns_by_item(cell_id, k + 1)[1:])
        knn = np.array(knn, dtype = int)
        distances = None

    else:

        if dist_metric == 'cosine':
            nbrs = NearestNeighbors(n_neighbors = k, metric = dist_metric, algorithm = 'brute').fit(embedding)
        else: nbrs = NearestNeighbors(n_neighbors = k, metric = dist_metric).fit(embedding)
        distances, knn = nbrs.kneighbors(return_distance = True)
    
    links = set([])
    for i in range(knn.shape[0]):
        for j in knn[i, :]:
            links.add(tuple(sorted((i, j))))

    return links, knn, distances


def adjacency_matrix(edges, n_nodes):
    
    A = scipy.sparse.lil_matrix((n_nodes, n_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    return A.tocsc()


def compute_neighbors(
    embedding, n_neighbors: int = 30, *, knn: bool = True, method = "umap",
    transformer = None, metric = "euclidean", metric_kwds = {},
    random_state = 0, n_jobs = -1
):
    
    n_cell, n_dim = embedding.shape
    if transformer is not None and not isinstance(transformer, str):
        n_neighbors = transformer.get_params()["n_neighbors"]
    
    # for very small dataset where there are less cells than requested knn.
    elif n_neighbors > n_cell:
        n_neighbors = 1 + int(0.5 * n_cell)
        warning(f"n_obs too small: adjusting to `n_neighbors = {n_neighbors}`")
    
    # default keyword arguments when `transformer` is not an instance
    method, transformer, shortcut = select_transformer(
        n_cell, method = method,  transformer = transformer, knn = knn, n_jobs = n_jobs, kwds = {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'metric_params': metric_kwds,
            'random_state': random_state
        }
    )

    if n_cell >= 10000 and not knn:
        warning("using such high n_obs without `knn = True` may take a lot of memory.")

    X = embedding
    distances = transformer.fit_transform(X)
    knn_indices, knn_distances = get_indices_distances_from_sparse_matrix(
        distances, n_neighbors
    )

    if shortcut:
        # self._distances is a sparse matrix with a diag of 1, fix that
        distances[np.diag_indices_from(distances)] = 0
        if knn:  # remove too far away entries in self._distances
            distances = get_sparse_matrix_from_indices_distances(
                knn_indices, knn_distances, keep_self=False)
        
        else:  # convert to dense
            distances = distances.toarray()
                
    if method == "umap":
        connectivities = umap_connectivity(
            knn_indices,
            knn_distances,
            n_obs = n_cell,
            n_neighbors= n_neighbors,
        )
    elif method == "gauss":
        connectivities = gauss_connectivity(
            distances, n_neighbors, knn = knn
        )

    elif method is not None:
        error('invalid method. should have been coerced.')

    connected_comps = None
    n_connected_comps = 1
    if scipy.sparse.issparse(connectivities):
        from scipy.sparse.csgraph import connected_components
        connected_comps = connected_components(connectivities)
        n_connected_comps = connected_comps[0]
    
    return (
        knn_indices, knn_distances, 
        distances, connectivities, 
        connected_comps, n_connected_comps
    )


def get_sparse_matrix_from_indices_distances(
    indices, distances, *, keep_self: bool,
) -> scipy.sparse.csr_matrix:
    """
    Create a sparse matrix from a pair of indices and distances.

    If keep_self = False, it verifies that the first column is the cell itself,
    then removes it from the explicitly stored zeroes.

    Duplicates in the data are kept as explicitly stored zeroes.
    """

    # instead of calling .eliminate_zeros() on our sparse matrix,
    # we manually handle the nearest neighbor being the cell itself.
    # this allows us to use _ind_dist_shortcut even when the data has duplicates.
    if not keep_self:
        indices, distances = remove_self_column(indices, distances)
    indptr = np.arange(0, np.prod(indices.shape) + 1, indices.shape[1])
    return scipy.sparse.csr_matrix(
        (
            # copy the data, otherwise strange behavior here
            distances.copy().ravel(),
            indices.copy().ravel(),
            indptr,
        ),
        shape=(indices.shape[0],) * 2,
    )


def select_transformer(
    n_obs, method, transformer, *, knn: bool, n_jobs, kwds
):
    """
    Return effective `method` and transformer. `method` will be coerced to `'gauss'` or `'umap'`.
    `transformer` is coerced from a str or instance to an instance class. If `transformer` is 
    `None` and there are few data points, `transformer` will be set to a brute force
    `sklearn.neighbors.KNeighborsTransformer`. If `transformer` is `None` and there are many 
    data points, `transformer` will be set like `umap` does (i.e. to a 
    `pynndescent.PyNNDescentTransformer` with custom `n_trees` and `n_iter`). This pynndescent
    implementation will return a much faster nn graph but with some approximations.
    """

    use_dense_distances = (
        kwds["metric"] == "euclidean" and n_obs < 8192
    ) or not knn

    shortcut = transformer == "sklearn" or (
        transformer is None and 
        (use_dense_distances or n_obs < 4096)
    )

    # validate `knn`
    conn_method = method if method in { "gauss", None } else "umap"
    if not knn and not (conn_method == "gauss" and transformer is None):
        # 'knn = False' seems to be only intended for method 'gauss'
        error('for method other than "gauss", you should set knn = True, since knn = False makes no sense.')
        
    # coerce `transformer` to an instance

    if shortcut:

        # for less than 4096 cells, you should just use brute force searcher.
        # this is not slow as for this scale.
        from sklearn.neighbors import KNeighborsTransformer
        assert transformer in { None, "sklearn" }

        n_neighbors = n_obs - 1
        if knn:  # only obey n_neighbors arg if knn set
            n_neighbors = min(n_neighbors, kwds["n_neighbors"])
        transformer = KNeighborsTransformer(
            algorithm = "brute",
            n_jobs = n_jobs,
            n_neighbors = n_neighbors,
            metric = kwds["metric"],
            metric_params = dict(kwds["metric_params"]),
            # no random_state
        )
    
    elif transformer is None or transformer == "pynndescent":
        from pynndescent import PyNNDescentTransformer
        kwds = kwds.copy()
        kwds["metric_kwds"] = kwds.pop("metric_params")
        if transformer is None:
            # Use defaults from UMAP’s `nearest_neighbors` function
            kwds.update(
                n_jobs = n_jobs,
                n_trees = min(64, 5 + int(round((n_obs) ** 0.5 / 20.0))),
                n_iters = max(5, int(round(np.log2(n_obs)))),
            )
        
        transformer = PyNNDescentTransformer(**kwds)

    elif isinstance(transformer, str):
        error(f'unknown transformer {transformer}.')

    # else `transformer` is probably an instance
    return conn_method, transformer, shortcut


def get_indices_distances_from_sparse_matrix(
    D: scipy.sparse.csr_matrix, n_neighbors: int
):
    """
    Get indices and distances from a sparse matrix.

    Makes sure that for both of the returned matrices:
    1. the first column corresponds to the cell itself as nearest neighbor.
    2. the number of neighbors (`.shape[1]`) is restricted to `n_neighbors`.
    """
    
    if (shortcut := index_distance_matrix_f(D)) is not None:
        indices, distances = shortcut
    else: indices, distances = index_distance_matrix_s(D, n_neighbors)

    # handle RAPIDS style indices_distances lacking the self-column
    if not has_self_column(indices, distances):
        indices = np.hstack([np.arange(indices.shape[0])[:, None], indices])
        distances = np.hstack([np.zeros(distances.shape[0])[:, None], distances])

    # if using the shortcut or adding the self column resulted in too many neighbors,
    # restrict the output matrices to the correct size
    if indices.shape[1] > n_neighbors:
        indices, distances = indices[:, :n_neighbors], distances[:, :n_neighbors]

    return indices, distances


def get_indices_distances_from_dense_matrix(D, n_neighbors: int):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances


def has_self_column(indices, distances) -> bool:
    # some algorithms have some messed up reordering.
    return (indices[:, 0] == np.arange(indices.shape[0])).any()


def remove_self_column(indices, distances):
    if not has_self_column(indices, distances):
        error("the first neighbor should be the cell itself.")
    return indices[:, 1:], distances[:, 1:]


def index_distance_matrix_s(D, n_neighbors: int):

    indices = np.zeros((D.shape[0], n_neighbors), dtype = int)
    distances = np.zeros((D.shape[0], n_neighbors), dtype = D.dtype)
    n_neighbors_m1 = n_neighbors - 1

    for i in range(indices.shape[0]):
        neighbors = D[i].nonzero()  # 'true' and 'spurious' zeros
        indices[i, 0] = i
        distances[i, 0] = 0

        # account for the fact that there might be more than n_neighbors
        # due to an approximate search
        # the point itself was not detected as its own neighbor during the search
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(D[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = D[i][
                neighbors[0][sorted_indices], neighbors[1][sorted_indices]
            ]

        else:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = D[i][neighbors]
    
    return indices, distances


def index_distance_matrix_f(D: scipy.sparse.csr_matrix):
    # check if each row has the correct number of entries
    nnzs = D.getnnz(axis = 1)
    if not is_constant(nnzs):
        warning("sparse matrix has no constant number of neighbors per row. ")
        warning("cannot efficiently get indices and distances.")
        return None
    
    n_obs, n_neighbors = D.shape[0], int(nnzs[0])
    return (
        D.indices.reshape(n_obs, n_neighbors),
        D.data.reshape(n_obs, n_neighbors),
    )


def gauss_connectivity(distances, n_neighbors: int, *, knn: bool):
    """
    Derive gaussian connectivities between data points from their distances.

    Parameters
    ----------
    distances
        The input matrix of distances between data points.

    n_neighbors
        The number of nearest neighbors to consider.

    knn
        Specify if the distances have been restricted to k nearest neighbors.
    """

    # init distances
    if isinstance(distances, scipy.sparse.csr_matrix):
        Dsq = distances.power(2)
        indices, distances_sq = get_indices_distances_from_sparse_matrix(
            Dsq, n_neighbors
        )
    
    else:
        assert isinstance(distances, np.ndarray)
        Dsq = np.power(distances, 2)
        indices, distances_sq = get_indices_distances_from_dense_matrix(
            Dsq, n_neighbors
        )

    # exclude the first point, the 0 th neighbor
    indices = indices[:, 1:]
    distances_sq = distances_sq[:, 1:]

    # choose sigma, the heuristic here doesn't seem to make much of a difference,
    # but is used to reproduce the figures of Haghverdi et al. (2016)
    if scipy.sparse.issparse(distances):
        # as the distances are not sorted
        # we have decay within the n_neighbors first neighbors
        sigmas_sq = np.median(distances_sq, axis=1)
    
    else:
        # the last item is already in its sorted position through argpartition
        # we have decay beyond the n_neighbors neighbors
        sigmas_sq = distances_sq[:, -1] / 4
    sigmas = np.sqrt(sigmas_sq)

    # compute the symmetric weight matrix
    if not scipy.sparse.issparse(distances):
        Num = 2 * np.multiply.outer(sigmas, sigmas)
        Den = np.add.outer(sigmas_sq, sigmas_sq)
        W = np.sqrt(Num / Den) * np.exp(-Dsq / Den)
        # make the weight matrix sparse
        if not knn:
            mask = W > 1e-14
            W[~ mask] = 0
        else:
            # restrict number of neighbors to ~ k
            # build a symmetric mask
            mask = np.zeros(Dsq.shape, dtype=bool)
            for i, row in enumerate(indices):
                mask[i, row] = True
                for j in row:
                    if i not in set(indices[j]):
                        W[j, i] = W[i, j]
                        mask[j, i] = True
            
            # set all entries that are not nearest neighbors to zero
            W[~ mask] = 0
    
    else:
        assert isinstance(Dsq, scipy.sparse.csr_matrix)
        W = Dsq.copy()  # need to copy the distance matrix here; what follows is inplace
        for i in range(len(Dsq.indptr[:-1])):
            row = Dsq.indices[Dsq.indptr[i] : Dsq.indptr[i + 1]]
            num = 2 * sigmas[i] * sigmas[row]
            den = sigmas_sq[i] + sigmas_sq[row]
            W.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] = np.sqrt(num / den) * np.exp(
                -Dsq.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] / den
            )

        W = W.tolil()
        for i, row in enumerate(indices):
            for j in row:
                if i not in set(indices[j]):
                    W[j, i] = W[i, j]
        W = W.tocsr()

    return W


def umap_connectivity(
    knn_indices, knn_dists, *,
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
) -> scipy.sparse.csr_matrix:
    """
    This is from umap.fuzzy_simplicial_set.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    import warnings
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore")
        from umap.umap_ import fuzzy_simplicial_set

    X = scipy.sparse.coo_matrix(([], ([], [])), shape = (n_obs, 1))
    connectivities, sigmas, rhos = fuzzy_simplicial_set(
        X, n_neighbors, None, None,
        knn_indices = knn_indices,
        knn_dists = knn_dists,
        set_op_mix_ratio = set_op_mix_ratio,
        local_connectivity = local_connectivity,
    )

    return connectivities.tocsr()
