
import numpy as np
import random

from exprmat.reduction.nn import knn_graph
from exprmat.utils import get_init_pos_from_paga, get_igraph_from_adjacency
from exprmat import error, warning, info


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


def draw_graph(
    adata,
    layout = "fa",
    *,
    init_pos: str | bool | None = None,
    paga_slot = 'paga',
    root: int | None = None,
    random_state = 0,
    adjacency = None,
    key_added_ext: str | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False, 
    use_gpu = False,
    **kwds,
):
    """
    Force-directed graph drawing.

    An alternative to tSNE that often preserves the topology of the data better. 
    This requires running `knn` first.

    The default layout ('fa', `ForceAtlas2`) uses the package `fa2-modified`,
    which can be installed via `pip install fa2-modified`.

    Force-directed graph drawing describes a class of long-established algorithms 
    for visualizing graphs. It was suggested for visualizing single-cell data by 
    :cite:t:`Islam2011`. Many other layouts as implemented in igraph :cite:p:`Csardi2006` 
    are available. Similar approaches have been used by :cite:t:`Zunder2015` or :cite:t:`Weinreb2017`.

    Parameters
    ----------
    adata
        Annotated data matrix.

    layout
        'fa' (`ForceAtlas2`) or any valid `igraph layout
        <https://igraph.org/c/doc/igraph-Layout.html>`__. Of particular interest
        are 'fr' (Fruchterman Reingold), 'grid_fr' (Grid Fruchterman Reingold,
        faster than 'fr'), 'kk' (Kamadi Kawai', slower than 'fr'), 'lgl' (Large
        Graph, very fast), 'drl' (Distributed Recursive Layout, pretty fast) and
        'rt' (Reingold Tilford tree layout).

    root
        Root for tree layouts.

    random_state
        For layouts with random initialization like 'fr', change this to use
        different intial states for the optimization. If `None`, no seed is set.

    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.

    key_added_ext
        By default, append `layout`.

    proceed
        Continue computation, starting off with 'X_draw_graph_`layout`'.

    init_pos
        `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
        Use precomputed coordinates for initialization. If `False`/`None` 
        (the default), initialize randomly.

    neighbors_key
        If not specified, draw_graph looks at .obsp['connectivities'] for connectivities
        (default storage place for pp.neighbors).
        If specified, draw_graph looks at .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
    
    obsp
        Use .obsp[obsp] as adjacency. You can't specify both
        `obsp` and `neighbors_key` at the same time.
        
    **kwds
        Parameters of chosen igraph layout. See e.g.
        :meth:`~igraph.GraphBase.layout_fruchterman_reingold`.
        One of the most important ones is `maxiter`.

    Returns
    -------
    .obsm['graph.[layout | key_added_ext]'] : numpy.ndarray (dtype `float`)
        Coordinates of graph layout.

    adata.uns['graph.[layout | key_added_ext]']: dict
        `draw_graph` parameters.
    """

    start = info(f"drawing single-cell graph using layout {layout!r}")
    adata = adata.copy() if copy else adata
    if adjacency is None:
        adjacency = adata.obsp[adata.uns[neighbors_key]['connectivities_key']]
    
    # init coordinates
    if init_pos in adata.obsm:
        init_coords = adata.obsm[init_pos]
    
    elif init_pos == "paga" or init_pos:
        init_coords = get_init_pos_from_paga(
            adata,
            adjacency,
            paga_slot = paga_slot,
            random_state = random_state,
            neighbors_key = neighbors_key,
            obsp = obsp,
        )

    else: # random initialization
        np.random.seed(random_state)
        init_coords = np.random.random((adjacency.shape[0], 2))

    if layout == 'fa' and use_gpu:
        draw_graph_gpu(adata, adjacency, init_coords)
        return
    
    # actual drawing
    if layout == "fa":
        positions = np.array(fa2_positions(adjacency, init_coords, **kwds))
    
    else:
        # igraph doesn't use numpy seed
        random.seed(random_state)

        g = get_igraph_from_adjacency(adjacency)
        if layout in {"fr", "drl", "kk", "grid_fr"}:
            ig_layout = g.layout(layout, seed = init_coords.tolist(), **kwds)
        
        elif "rt" in layout:
            if root is not None: root = [root]
            ig_layout = g.layout(layout, root=root, **kwds)
        
        else: ig_layout = g.layout(layout, **kwds)
        positions = np.array(ig_layout.coords)
    
    key_added = f"graph.{key_added_ext or layout}"
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(layout = layout, random_state = random_state)
    adata.obsm[key_added] = positions
    return adata if copy else None


def fa2_positions(
    adjacency: np.ndarray, init_coords: np.ndarray, **kwds
) -> list[tuple[float, float]]:
    
    from fa2_modified import ForceAtlas2 as fa2

    forceatlas2 = fa2(
    
        # behavior alternatives
        outboundAttractionDistribution = False,  # dissuade hubs
        linLogMode = False,   # not implemented
        adjustSizes = False,  # not implemented
        edgeWeightInfluence = 1.0,

        # performance
        jitterTolerance = 1.0,  # tolerance
        barnesHutOptimize = True,
        barnesHutTheta = 1.2,
        multiThreaded = False,  # not implemented
        
        # tuning
        scalingRatio = 2.0,
        strongGravityMode = False,
        gravity = 1.0,
        verbose = False
    )

    if "maxiter" in kwds:
        iterations = kwds["maxiter"]
    elif "iterations" in kwds:
        iterations = kwds["iterations"]
    else: iterations = 500

    return forceatlas2.forceatlas2(
        adjacency, 
        pos = init_coords, 
        iterations = iterations
    )


def draw_graph_gpu(
    adata, adjacency, init_coords = None, max_iter: int = 500
) -> None:
    
    from cugraph import Graph
    from cugraph.layout import force_atlas2 as fa2
    import cudf 
    import cupy as cp

    # adjacency graph
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = Graph()
    if hasattr(g, "add_adj_list"): g.add_adj_list(offsets, indices, None)
    else: g.from_cudf_adjlist(offsets, indices, None)

    if init_coords is not None:
        x, y = np.hsplit(init_coords, init_coords.shape[1])
        inital_df = cudf.DataFrame({"x": x.ravel(), "y": y.ravel()})
        inital_df["vertex"] = inital_df.index
    else: inital_df = None

    # run cugraphs Force Atlas 2
    positions = fa2(
        input_graph = g,
        pos_list = inital_df,
        max_iter = max_iter,
        outbound_attraction_distribution = False,
        lin_log_mode = False,
        edge_weight_influence = 1.0,
        
        jitter_tolerance = 1.0,
        barnes_hut_optimize = True,
        barnes_hut_theta = 1.2,
        
        scaling_ratio = 2.0,
        strong_gravity_mode = False,
        gravity = 1.0,
    )

    positions = cp.vstack((positions["x"].to_cupy(), positions["y"].to_cupy())).T
    layout = "fa"
    adata.uns["graph.fa"] = {}
    adata.uns["graph.fa"]["params"] = { "layout": layout, "random_state": 0, 'gpu': True }
    key_added = f"graph.{layout}"
    adata.obsm[key_added] = positions.get()