
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
import scipy.sparse

from exprmat.ansi import warning, info


def knn_graph(
        embedding, k = 5, dist_metric = 'euclidean', approx = False, 
        random_seed = 42
    ):
    '''
    Calculate exact or approximate kNN graph.
    Using approximate kNN requires installation of package ``annoy``.

    Returns
    ----------

    A tuple of three:
    
    *   edges of the graph (n_edges, 2)
    *   nearest neighbor index matrix (n_cells, n_neighbors)
    *   distance matrix (n_cells, n_cells)
    '''

    if approx:
        try: from annoy import AnnoyIndex
        except:
            approx = False
            warning(f'approximate nn graph requires installation of `annoy`.')
            warning(f'fall back to brute force neighbors retrieval.')
    
    if approx:
        
        if dist_metric == 'cosine': dist_metric = 'angular'
        npc = embedding.shape[1]
        ncell = embedding.shape[0]
        annoy_index = AnnoyIndex(npc, metric = dist_metric)
        annoy_index.set_seed(random_seed)

        for i in range(ncell):
            annoy_index.add_item(i, list(embedding[i,:]))
        annoy_index.build(10) # 10 trees

        knn = []
        for cell_id in range(ncell):
            knn.append(annoy_index.get_nns_by_item(cell_id, k + 1)[1:])
        knn = np.array(knn, dtype = int)
        distances = None

    else:

        if dist_metric == 'cosine':
            nbrs = NearestNeighbors(n_neighbors = k, metric = dist_metric, algorithm = 'brute').fit(embedding)
        else: nbrs = NearestNeighbors(n_neighbors = k, metric = dist_metric).fit(embedding)
        distances, knn = nbrs.kneighbors(return_distance = True)

    
    links = set([])
    for i in range(knn.shape[0]):
        for j in knn[i, :]:
            links.add(tuple(sorted((i, j))))

    return links, knn, distances


def adjacency_matrix(edges, n_nodes):
    A = scipy.sparse.lil_matrix((n_nodes, n_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    return A.tocsc()