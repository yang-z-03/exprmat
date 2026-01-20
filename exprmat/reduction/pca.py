
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA


def pca(
    S, cell_ids = [], n_comp = 50, keep_sparse = False, 
    random_state = 42, svd_solver = 'arpack'
):
    '''
    Run PCA on the scaled matrix S. If keep_sparse is set to True, truncated SVD is
    used to preserve the sparsitdy in calculation.

    Returns
    ------------
    
    A tuple of 7:

    *   The transformed embedding by the PCA embedder. (n_cells, n_comp)
    *   The components of PCA. (n_comp, n_genes)
    *   Explained variance
    *   Percentage of explained variance
    *   Singular values lambda of the components
    *   Model parameters, used to store in the operation metadata.
    *   The PCA embedder
    '''
    
    if len(cell_ids) == 0:
        cell_ids = np.arange(S.shape[0])

    if keep_sparse:
        pca = TruncatedSVD(
            n_components = n_comp, 
            random_state = random_state, 
            algorithm = svd_solver
        )

    else:
        pca = PCA(
            n_components = n_comp, 
            random_state = random_state, 
            svd_solver = svd_solver
        )

    pca = pca.fit(S[cell_ids, :])
    return (
        pca.transform(S),
        pca.components_,
        pca.explained_variance_,
        pca.explained_variance_ratio_, 
        pca.singular_values_,
        pca.get_params(),
        pca
    )