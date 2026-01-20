
import scipy.sparse as ss
import numpy as np

import exprmat.snapatac as internal
from exprmat.utils import get_igraph_from_adjacency, is_anndata 


def kmeans(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    n_clusters: int,
    n_iterations: int = -1,
    random_state: int = 0,
    use_rep: str = "spectral",
    key_added: str = 'kmeans',
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Cluster cells into subgroups using the K-means algorithm.

    Parameters
    ----------
    n_clusters
        Number of clusters to return. Note that unlike other methods, kmeans requires
        a predefined number of clusters expected. This should be manually examined, using
        tools like silhoutte metrics and ROGUE homogeneity tests.

    n_iterations
        How many iterations of the kmeans clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
    """

    import polars

    if is_anndata(adata): data = adata.obsm[use_rep]
    else: data = adata

    groups = internal.kmeans(n_clusters, data)
    groups = np.array(groups, dtype = np.str_)

    if inplace:
        adata.obs[key_added] = polars.Series(
            groups, dtype = polars.datatypes.Categorical,
        )

    else: return groups
