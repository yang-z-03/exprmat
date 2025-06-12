
from exprmat.dynamics.preprocess import filter_and_normalize
from exprmat.dynamics.pseudotime import velocity_pseudotime, terminal_states
from exprmat.dynamics.transition import transition_matrix
from exprmat.dynamics.moments import moments
from exprmat.dynamics.velocity import velocity, velocity_confidence, velocity_embedding, velocity_graph
from exprmat.dynamics.generank import rank_velocity_genes


def run_velocity(
    adata, 
    neighbor_key = 'neighbors', neighbor_connectivity = 'connectivities', n_neighbors = 35, 
    hvg = 'vst.norm', 
    velocity_key = 'velocity',
    n_cpus = None,
    kwargs_filter = { 'enforce': False },
    kwargs_velocity = {},
    kwargs_velocity_graph = {},
    kwargs_terminal_state = {},
    kwargs_pseudotime = { 'save_diffmap': True },
):
    if (
        ('spliced.counts' not in adata.layers.keys()) and 
        ('unspliced.counts' not in adata.layers.keys())
    ):
        filter_and_normalize(adata, **kwargs_filter)
    
    moments(adata, neighbor_key = neighbor_key, mode = neighbor_connectivity)
    
    velocity(
        adata, 
        vkey = velocity_key, 
        use_highly_variable = hvg, 
        neighbor_key = neighbor_key, 
        **kwargs_velocity
    )

    velocity_graph(
        adata, 
        n_jobs = n_cpus, 
        vkey = velocity_key,
        neighbor_key = neighbor_key,
        mode_neighbors = neighbor_connectivity,
        n_neighbors = n_neighbors,
        **kwargs_velocity_graph
    )

    terminal_states(adata, vkey = velocity_key, neighbor_key = neighbor_key, **kwargs_terminal_state)
    velocity_pseudotime(adata, vkey = velocity_key, neighbor_key = neighbor_key, **kwargs_pseudotime)
    velocity_confidence(adata, vkey = velocity_key, neighbor_key = neighbor_key)
    