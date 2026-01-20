
def get_spectral_clusters(A, k):
    
    from sklearn.cluster import SpectralClustering
    spec = SpectralClustering(
        n_clusters = k, random_state = 0, 
        affinity = 'precomputed', assign_labels = 'discretize'
    )

    return spec.fit_predict(A)