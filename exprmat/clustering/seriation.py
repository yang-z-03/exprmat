
import numpy as np
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage


def seriation(Z, N, cur_index):
    '''
    Computes the order implied by a hierarchical tree (dendrogram)

    Parameters
    ----------
    Z:
        A hierarchical tree (dendrogram)
    
    N:
        The number of points given to the clustering process

    cur_index:
        The position in the tree for the recursive traversal
    
    Returns
    -------
    Order implied by the hierarchical tree Z
    '''

    if cur_index < N: return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))
    

def serial_matrix(dist, method = "ward", seed = 42):
    '''
    Compute_serial_matrix transforms a distance matrix into a sorted distance matrix 
    according to the order implied by the hierarchical tree (dendrogram).

    Parameters
    ----------
    dist:
        Distance matrix
    
    method: ["ward", "single", "average", "complete"]
        Method of agglomerative clustering

    Returns
    -------
    - seriated_dist: The input dist_mat, but with re-ordered rows and columns
      according to the seriation, i.e. the order implied by the hierarchical tree
    - res_order: The order implied by the hierarhical tree
    - res_linkage: The hierarhical tree (dendrogram)
    '''
    np.random.seed(seed)

    N = len(dist)
    flat_dist_mat = squareform(dist)
    res_linkage = linkage(flat_dist_mat, method = method,preserve_input = True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N, k = 1)
    seriated_dist[a,b] = dist[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage