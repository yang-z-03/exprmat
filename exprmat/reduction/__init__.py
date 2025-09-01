
import anndata as ad
from exprmat.ansi import error
from sklearn.utils import check_random_state
from exprmat.utils import choose_representation, choose_layer
from exprmat.reduction.umap import umap_simplical
from exprmat.reduction.pca import pca
from exprmat.reduction.nn import compute_neighbors


def run_pca(
    adata, *, layer = 'scaled', n_comps = 50, keep_sparse = False, 
    random_state = 42, svd_solver = 'arpack', key_added = 'pca',
    use_gpu = False
):
    if use_gpu:
        
        # we do not encourage shipping such a big, condense matrix (scaled data)
        # onto the gpu. instead we prefer using models that utilizes gpu directly
        # from sparse data (e.g. scVI)

        if adata.shape[0] * adata.shape[1] >= 1024 * 1024 * 1024:
            from exprmat.utils import configure_vram_flavor
            configure_vram_flavor('ram')

        import rapids_singlecell as rsc
        import anndata as ad
        import cupyx as cpx

        S = choose_layer(adata, layer = layer)
        gpu_adata = ad.AnnData(X = S)
        rsc.get.anndata_to_GPU(gpu_adata)
        
        rsc.pp.pca(
            adata = gpu_adata, n_comps = n_comps, layer = None,
            svd_solver = None, key_added = 'pca'
        )

        # copy to adata
        adata.obsm[key_added] = gpu_adata.obsm['pca']
        adata.varm[key_added] = gpu_adata.varm['pca']
        adata.uns[key_added] = {
            'variance': gpu_adata.uns['pca']['variance'],
            'pct.variance': gpu_adata.uns['pca']['variance_ratio'],
            'singular': None,
            'params': {
                'n_comps': n_comps,
                'gpu': True
            }
        }

    else:
        S = choose_layer(adata, layer = layer)
        emb, comps, variance, pct_variance, singular, params, model = pca(
            S, n_comp = n_comps, keep_sparse = keep_sparse, 
            random_state = random_state, svd_solver = svd_solver
        )

        adata.obsm[key_added] = emb
        adata.varm[key_added] = comps.T
        adata.uns[key_added] = {
            'variance': variance,
            'pct.variance': pct_variance,
            'singular': singular,
            'params': params
        }

    return


def run_knn(
    adata, *, use_rep = 'pca', n_comps = None,
    n_neighbors: int = 30, knn: bool = True, method = "umap",
    transformer = None, metric = "euclidean", metric_kwds = {},
    random_state = 0, key_added = 'neighbors', n_jobs = -1,

    # only used in gpu accelerated version
    gpu_approx_method = 'nn_descent', gpu_approx_method_kwds = {},
    use_gpu = False
):
    emb = choose_representation(adata, use_rep = use_rep, n_pcs = n_comps)

    if use_gpu:

        # compute a neighborhood graph of observations with cuml.
        # the neighbor search efficiency of this heavily relies on cuml, which also 
        # provides a method for estimating connectivities of data points - the 
        # connectivity of the manifold.

        # create a pseudo adata
        import rapids_singlecell as rsc
        import anndata as ad
        import cupyx.scipy.sparse as sp

        emb_data = ad.AnnData(X = sp.csr_matrix(adata.shape, dtype = 'float32'))
        emb_data.obsm['emb'] = emb

        rsc.pp.neighbors(
            adata = emb_data, n_neighbors = n_neighbors, n_pcs = None,
            use_rep = 'emb', random_state = random_state,
            algorithm = gpu_approx_method, metric = metric,
            metric_kwds = metric_kwds, key_added = None,
            algorithm_kwds = gpu_approx_method_kwds
        )

        adata.uns[key_added] = {
            'connectivities_key': 'connectivities' if key_added == 'neighbors' else 'connectivities.' + key_added,
            'neighbors_key': 'neighbors' if key_added == 'neighbors' else 'neighbors.' + key_added,
            'distances_key': 'distances' if key_added == 'neighbors' else 'distances.' + key_added,
            'knn_key': 'knn' if key_added == 'neighbors' else 'knn.' + key_added,
            'knn_distances_key': 'knn.d' if key_added == 'neighbors' else 'knn.d.' + key_added,
            'metric': metric,
            'metric_kwds': metric_kwds,
            'random_state': random_state,
            'knn': knn,
            'method': method,
            'n_neighbors': n_neighbors,
            'use_rep': use_rep,
            'n_comps': n_comps if n_comps is not None else emb.shape[1],
            'gpu': True,
            'gpu_approx_method': gpu_approx_method,
            'gpu_approx_method_kwds': gpu_approx_method_kwds
        }

        adata.obsp[adata.uns[key_added]['connectivities_key']] = emb_data.obsp['connectivities']
        adata.obsp[adata.uns[key_added]['distances_key']] = emb_data.obsp['distances']
        
        from exprmat.reduction.nn import get_indices_distances_from_sparse_matrix
        knn_indices, knn_dist = get_indices_distances_from_sparse_matrix(
            emb_data.obsp['distances'], n_neighbors
        )

        adata.obsm[adata.uns[key_added]['knn_key']] = knn_indices
        adata.obsm[adata.uns[key_added]['knn_distances_key']] = knn_dist

    else:
        knn_indices, knn_dist, dist, conn, connected_comp, n_conn_comp = compute_neighbors(
            emb, n_neighbors = n_neighbors, knn = knn, method = method,
            transformer = transformer, metric = metric, metric_kwds = metric_kwds,
            random_state = random_state, n_jobs = n_jobs
        )

        adata.uns[key_added] = {
            'connectivities_key': 'connectivities' if key_added == 'neighbors' else 'connectivities.' + key_added,
            'neighbors_key': 'neighbors' if key_added == 'neighbors' else 'neighbors.' + key_added,
            'distances_key': 'distances' if key_added == 'neighbors' else 'distances.' + key_added,
            'knn_key': 'knn' if key_added == 'neighbors' else 'knn.' + key_added,
            'knn_distances_key': 'knn.d' if key_added == 'neighbors' else 'knn.d.' + key_added,
            'metric': metric,
            'metric_kwds': metric_kwds,
            'random_state': random_state,
            'knn': knn,
            'method': method,
            'n_neighbors': n_neighbors,
            'use_rep': use_rep,
            'n_comps': n_comps if n_comps is not None else emb.shape[1]
        }

        adata.obsp[adata.uns[key_added]['connectivities_key']] = conn
        adata.obsp[adata.uns[key_added]['distances_key']] = dist
        adata.obsm[adata.uns[key_added]['knn_key']] = knn_indices
        adata.obsm[adata.uns[key_added]['knn_distances_key']] = knn_dist
    return


def run_umap(
    adata, *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos = "spectral",
    random_state = 0,
    a: float | None = None,
    b: float | None = None,
    key_added: str | None = None,
    neighbors_key: str = "neighbors",
    use_gpu = False
):

    if use_gpu:

        from exprmat.reduction.umap import umap_gpu
        umap_gpu(
            adata, 
            min_dist = min_dist,
            spread = spread,
            n_components = n_components,
            maxiter = maxiter,
            alpha = alpha,
            negative_sample_rate = negative_sample_rate,
            init_pos = init_pos,
            random_state = random_state,
            a = a,
            b = b,
            key_added = key_added,
            neighbors_key = neighbors_key
        )

        return
    
    key_obsm, key_uns = ("umap", "umap") if key_added is None else [key_added] * 2
    if neighbors_key is None:
        neighbors_key = "neighbors"
    if neighbors_key not in adata.uns.keys():
        error(f'do not find the key `{neighbors_key}` in uns slots')
    from umap.umap_ import find_ab_params, simplicial_set_embedding

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    adata.uns[key_uns] = dict(params = dict(a = a, b = b))

    if (
        isinstance(init_pos, str) and 
        (init_pos in adata.obsm) and
        (adata.obsm[init_pos].shape[1] == n_components)
    ):
        init_coords = adata.obsm[init_pos]
    else: init_coords = init_pos # let umap handle it

    if random_state != 0:
        adata.uns[key_uns]["params"]["random_state"] = random_state
    random_state = check_random_state(random_state)

    neigh_params = adata.uns[neighbors_key]
    X = choose_representation(
        adata, use_rep = neigh_params.get("use_rep", None),
        n_pcs = neigh_params.get("n_pcs", None), silent = True,
    )

    conn_key = neigh_params.get("connectivities_key", "connectivities")
    if conn_key not in adata.obsp.keys():
        error(f'do not find neighbor connectivity key `{conn_key}`')
    connectivities = adata.obsp[conn_key]
    
    # the data matrix X is really only used for determining the number of connected components
    # for the init condition in the UMAP embedding

    default_epochs = 500 if connectivities.shape[0] <= 10000 else 200
    n_epochs = default_epochs if maxiter is None else maxiter
    
    emb = umap_simplical(
        X, connectivities, init_coords,
        n_components = n_components,
        epoch = n_epochs,
        alpha = alpha, gamma = gamma, negative_sample_rate = negative_sample_rate,
        random_state = random_state, a = a, b = b,
        metric = neigh_params.get("metric", "euclidean"),
        metric_kwds = neigh_params.get("metric_kwds", {})
    )

    adata.obsm[key_obsm] = emb
    return