
import numpy as np
from exprmat.reduction.nn import knn_graph


def fa2(X, n_neighbors = 5, approx_neighbors = False, n_iter = 300, verbose = False):
    edges, _, _ = knn_graph(X, k = n_neighbors, approx = approx_neighbors)
    return layout(edges, X.shape[0], verbose = verbose)


def layout(
    edges, n_cells, n_iter = 100, edge_weight_influence = 1,
    barnes_hut_theta = 2, scale_ratio = 1, gravity = 0.05,
    jitter_tolerance = 1, verbose = False
):
    from fa2_modified import ForceAtlas2 as fatlas
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n_cells))
    G.add_edges_from(list(edges))

    forceatlas2 = fatlas(
        # behavior alternatives
        outboundAttractionDistribution = False,
        edgeWeightInfluence = edge_weight_influence,
        # performance
        jitterTolerance = jitter_tolerance,
        barnesHutOptimize = True,
        barnesHutTheta = barnes_hut_theta,
        # tuning
        scalingRatio = scale_ratio,
        strongGravityMode = False,
        gravity = gravity,
        # log
        verbose = verbose,
    )

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos = None, iterations = n_iter)
    positions = np.array([positions[i] for i in sorted(positions.keys())])
    return positions
