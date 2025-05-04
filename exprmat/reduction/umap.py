
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