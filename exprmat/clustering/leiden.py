
import anndata as ad
import scanpy as sc


def leiden(
    adata: ad.AnnData,
    resolution: float = 1,
    *,
    restrict_to = None,
    random_state = 0,
    key_added: str = 'leiden',
    adjacency = None,
    directed: bool | None = None,
    use_weights: bool = True,
    n_iterations: int = 2,
    partition_type = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    flavor = 'igraph',
    ** clustering_args,
) -> ad.AnnData | None:
    '''
    Cluster cells using the Leiden algorithm, an improved version of the Louvain algorithm.

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        A parameter value controlling the coarseness of the clustering. Higher values lead 
        to more clusters. Set to `None` if overriding `partition_type` to one that doesn't 
        accept a `resolution_parameter`.

    random_state
        Change the initialization of the optimization.

    restrict_to
        Restrict the clustering to the categories within the key for sample annotation, 
        tuple needs to contain `(obs_key, list_of_categories)`.

    key_added
        `adata.obs` key under which to add the cluster labels.

    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.

    directed
        Whether to treat the graph as directed or undirected. `igrah` implementation
        does not support a directed graph, you will need to drop back to `leidenalg`
        method for directed neighbor graphs

    use_weights
        If `True`, edge weights from the graph are used in the computation.
        (placing more emphasis on stronger edges).

    n_iterations
        How many iterations of the Leiden clustering algorithm to perform. Positive values 
        above 2 define the total number of iterations to perform, -1 has the algorithm run 
        until it reaches its optimal clustering. 2 is faster and the default for 
        underlying packages.

    partition_type
        Type of partition to use. Defaults to `leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for `leidenalg.find_partition`.
    
    neighbors_key
        Use neighbors connectivities as adjacency. If not specified, leiden looks at 
        .obsp['connectivities'] for connectivities (default storage place for pp.neighbors).
        If specified, leiden looks at .obsp[.uns[neighbors_key]['connectivities_key']] for 
        connectivities.

    obsp
        Use .obsp[obsp] as adjacency. You can't specify both `obsp` and `neighbors_key` 
        at the same time.

    copy
        Whether to copy `adata` or modify it inplace.

    flavor
        Which package's implementation to use.

    clustering_args
        Any further arguments to pass to :func:`~leidenalg.find_partition` (which in turn 
        passes arguments to the `partition_type`) or `igraph.Graph.community_leiden` from `igraph`.
    '''
    
    return sc.tl.leiden(
        adata, resolution = resolution,
        restrict_to = restrict_to, random_state = random_state,
        key_added = key_added, adjacency = adjacency, directed = directed,
        use_weights = use_weights, n_iterations = n_iterations,
        partition_type = partition_type, neighbors_key = neighbors_key,
        copy = False, flavor = flavor, **clustering_args
    )