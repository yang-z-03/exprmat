
from __future__ import annotations

from typing import Literal
import scipy.sparse as ss
import numpy as np

import exprmat.snapatac as internal
from exprmat.utils import get_igraph_from_adjacency, is_anndata 


def hdbscan(
    adata: internal.AnnData,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    cluster_selection_method: str = "eom",
    random_state: int = 0,
    use_rep: str = "spectral",
    key_added: str = 'hdbscan',
    ** kwargs,
) -> None:
    """
    Cluster cells into subgroups using the HDBSCAN algorithm.

    Parameters
    ----------
    min_cluster_size
        The minimum size of clusters; single linkage splits that contain fewer points 
        than this will be considered points "falling out" of a cluster rather than a 
        cluster splitting into two new clusters.

    min_samples
        The number of samples in a neighbourhood for a point to be considered a core point.

    cluster_selection_epsilon
        A distance threshold. Clusters below this value will be merged.

    alpha
        A distance scaling parameter as used in robust single linkage.

    cluster_selection_method
        The method used to select clusters from the condensed tree. The standard approach 
        for HDBSCAN* is to use an Excess of Mass algorithm to find the most persistent clusters.
        Alternatively you can instead select the clusters at the leaves of the tree - this 
        provides the most fine grained and homogeneous clusters. Options are: "eom" or "leaf".

    random_state
        Change the initialization of the optimization.

    """

    import pandas as pd
    import hdbscan

    data = adata.obsm[use_rep][...]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples = min_samples,
        cluster_selection_epsilon = cluster_selection_epsilon,
        alpha = alpha,
        cluster_selection_method = cluster_selection_method,
        **kwargs
    )

    clusterer.fit(data)
    groups = clusterer.labels_
    adata.obs[key_added] = pd.Categorical(
        values = groups.astype('U'),
        categories = sorted(map(str, np.unique(groups))),
    )


def dbscan(
    adata: internal.AnnData,
    eps: float = 0.5,
    min_samples: int = 5,
    leaf_size: int = 30,
    n_jobs: int | None = None,
    use_rep: str = "spectral",
    key_added: str = 'dbscan',
) -> None:
    """
    Cluster cells into subgroups using the DBSCAN algorithm.

    Parameters
    ----------
    eps
        The maximum distance between two samples for one to be considered as in the neighborhood 
        of the other. This is not a maximum bound on the distances of points within a cluster. 
        This is the most important DBSCAN parameter to choose appropriately for your data set 
        and distance function.
    
    min_samples
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    
    leaf_size
        Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction 
        and query, as well as the memory required to store the tree. The optimal value depends 
        on the nature of the problem.
    
    n_jobs
        The number of parallel jobs to run. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    """

    from sklearn.cluster import DBSCAN
    import pandas as pd

    data = adata.obsm[use_rep][...]

    clustering = DBSCAN(
        eps = eps,
        min_samples = min_samples,
        metric = 'euclidean',
        leaf_size = leaf_size,
        n_jobs = n_jobs
    ).fit(data)

    groups = clustering.labels_
    adata.obs[key_added] = pd.Categorical(
        values = groups.astype('U'),
        categories=sorted(map(str, np.unique(groups))),
    )