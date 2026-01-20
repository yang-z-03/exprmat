
def tsne(X, angle = 0.5, perplexity = 30, random_state = 42, verbose = False):
    
    from sklearn.manifold import TSNE
    embedder = TSNE(
        angle = angle, 
        perplexity = perplexity, 
        random_state = random_state, 
        verbose = verbose
    )

    return embedder.fit_transform(X)
