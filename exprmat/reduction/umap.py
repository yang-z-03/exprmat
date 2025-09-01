
from sklearn.utils import check_random_state


def umap(
    embedding, n_neighbors = 10, min_dist = 0.1, 
    metric = 'euclidean', random_state = 42
):

    from umap import UMAP as umap_model
    embedder = umap_model(
        n_neighbors = n_neighbors, 
        min_dist = min_dist, 
        metric = metric, 
        random_state = random_state
    )
    
    embedder.fit(embedding)
    return embedder, embedder.transform(embedding)


def umap_simplical(
    X, connectivities, init_coords, *,
    n_components: int = 2,
    epoch: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    random_state = 0,
    a: float | None = None,
    b: float | None = None,
    metric = 'euclidean', metric_kwds = {}
):
    from umap.umap_ import simplicial_set_embedding
    
    emb, _ = simplicial_set_embedding(
        data = X,
        graph = connectivities.tocoo(),
        n_components = n_components,
        initial_alpha = alpha,
        a = a, b = b,
        gamma = gamma,
        negative_sample_rate = negative_sample_rate,
        n_epochs = epoch,
        init = init_coords,
        random_state = random_state,
        metric = metric,
        metric_kwds = metric_kwds,
        densmap = False,
        densmap_kwds = {},
        output_dens = False,
        verbose = False,
    )

    return emb


def umap_gpu(
    adata,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos = "auto",
    random_state: int = 0,
    a: float | None = None,
    b: float | None = None,
    key_added: str | None = None,
    neighbors_key: str | None = None,
    copy: bool = False,
):
    
    import cuml
    import cuml.internals.logger as logger
    import cupy as cp
    import numpy as np
    from cuml.manifold.simpl_set import simplicial_set_embedding
    from cuml.manifold.umap import UMAP
    from cuml.manifold.umap_utils import find_ab_params
    from cuml.thirdparty_adapters import check_array as check_array_cuml
    from cupyx.scipy import sparse
    from packaging.version import parse as parse_version
    from scanpy._utils import NeighborsView
    from scanpy.tools._utils import get_init_pos_from_paga
    from sklearn.utils import check_random_state
    from exprmat import error
    
    adata = adata.copy() if copy else adata

    if neighbors_key is None:
        neighbors_key = "neighbors"

    if neighbors_key not in adata.uns:
        error(f'did not find .uns["{neighbors_key}"]. Run `knn` first.')

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)

    # store params for adata.uns
    stored_params = {
        "a": a,
        "b": b,
        **({"random_state": random_state} if random_state != 0 else {}),
    }

    neigh_params = adata.uns[neighbors_key]
    X = adata.obsm[neigh_params['use_rep']]

    n_epochs = (
        500 if maxiter is None else maxiter
    )  # 0 is not a valid value for rapids, unlike original umap

    n_obs = adata.shape[0]
    if parse_version(cuml.__version__) < parse_version("24.10"):
        # `simplicial_set_embedding` is bugged in cuml<24.10. This is why we use `UMAP` instead.
        n_neighbors = neigh_params["n_neighbors"]
        if neigh_params.get("method") == "rapids":
            knn_dist = adata.obsp[neigh_params['distances_key']].data.reshape(n_obs, n_neighbors)
            knn_indices = adata.obsp[neigh_params['distances_key']].indices.reshape(n_obs, n_neighbors)
            pre_knn = (knn_indices, knn_dist)
        else:
            pre_knn = None

        if init_pos not in ["auto", "spectral", "random"]:
            error("valid init_pos are: auto, spectral, random, paga for RAPIDS < 24.10",)

        random_state = check_random_state(random_state)

        if init_pos == "auto":
            init_pos = "spectral" if n_obs < 1000000 else "random"

        umap = UMAP(
            n_neighbors = n_neighbors,
            n_components = n_components,
            metric = neigh_params.get("metric", "euclidean"),
            metric_kwds = neigh_params.get("metric_kwds", None),
            n_epochs = n_epochs,
            learning_rate = alpha,
            init = init_pos,
            min_dist = min_dist,
            spread = spread,
            negative_sample_rate = negative_sample_rate,
            a = a,
            b = b,
            random_state = random_state,
            output_type = "numpy",
            precomputed_knn = pre_knn,
        )

        X_umap = umap.fit_transform(X)

    else:

        pre_knn = adata.obsp[neigh_params['connectivities_key']]

        match init_pos:
            case str() if init_pos in adata.obsm:
                init_coords = adata.obsm[init_pos]
            case str() if init_pos == "auto":
                init_coords = "spectral" if n_obs < 1000000 else "random"
            case _: init_coords = init_pos

        if hasattr(init_coords, "dtype"):
            init_coords = check_array_cuml(
                init_coords, dtype = np.float32, accept_sparse = False
            )

        random_state = check_random_state(random_state)

        X_umap = simplicial_set_embedding(
            data = cp.array(X),
            graph = sparse.coo_matrix(pre_knn),
            n_components = n_components,
            initial_alpha = alpha,
            a = a,
            b = b,
            negative_sample_rate = negative_sample_rate,
            n_epochs = n_epochs,
            init = init_coords,
            random_state = random_state,
            metric = neigh_params.get("metric", "euclidean"),
            metric_kwds = neigh_params.get("metric_kwds", None),
        )
        
        X_umap = cp.asarray(X_umap).get()

    key_obsm, key_uns = ("X_umap", "umap") if key_added is None else [key_added] * 2
    adata.obsm[key_obsm] = X_umap

    adata.uns[key_uns] = stored_params
    return adata if copy else None