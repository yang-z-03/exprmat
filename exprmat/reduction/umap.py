
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
