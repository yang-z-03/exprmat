"""
Personalized PageRank (PPR) algorithm implementation
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Union
import networkx as nx


def personalized_pagerank(
    graph: nx.DiGraph,
    seed_nodes: Union[str, List[str]],
    damping_factor: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Calculate personalized PageRank scores for seed nodes.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        Directed graph with edge weights
    seed_nodes : str or list of str
        Seed node(s) for personalized PageRank
    damping_factor : float
        Probability of continuing walk (default 0.5)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    np.ndarray
        Personalized PageRank scores for all nodes
    """
    if isinstance(seed_nodes, str):
        seed_nodes = [seed_nodes]
    
    # Get all nodes
    nodes = list(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize preference vector (E)
    E = np.zeros(n)
    for seed in seed_nodes:
        if seed in node_to_idx:
            E[node_to_idx[seed]] = 1.0 / len(seed_nodes)
    
    # Build transition matrix
    # Get adjacency matrix with weights
    adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes, weight='weight')
    
    # Normalize by out-degree (column normalization for transpose)
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1  # Avoid division by zero
    
    # Create normalized transition matrix
    # For PPR, we use column normalization (transpose of row-normalized)
    transition = adj_matrix.T
    transition = transition.multiply(1.0 / out_degree)
    
    # Convert to dense for iteration (can be optimized with sparse operations)
    transition = transition.toarray()
    
    # Initialize PageRank vector
    pagerank = np.ones(n) / n
    
    # Iterate until convergence
    for iteration in range(max_iter):
        pagerank_new = (1 - damping_factor) * E + damping_factor * transition.dot(pagerank)
        
        # Check convergence
        if np.linalg.norm(pagerank_new - pagerank, ord=1) < tol:
            break
        
        pagerank = pagerank_new
    
    return pagerank


def pagerank_from_edges(
    edges: np.ndarray,
    weights: np.ndarray,
    node_list: List[str],
    seed_nodes: Union[str, List[str]],
    damping_factor: float = 0.5
) -> np.ndarray:
    """
    Calculate personalized PageRank from edge list.
    
    Parameters:
    -----------
    edges : np.ndarray
        Array of shape (n_edges, 2) with node indices
    weights : np.ndarray
        Edge weights
    node_list : list of str
        List of all node names
    seed_nodes : str or list of str
        Seed node(s) for personalized PageRank
    damping_factor : float
        Probability of continuing walk
    
    Returns:
    --------
    np.ndarray
        Personalized PageRank scores
    """
    # Create graph from edges
    graph = nx.DiGraph()
    
    # Add edges with weights
    for (from_idx, to_idx), weight in zip(edges, weights):
        from_node = node_list[from_idx]
        to_node = node_list[to_idx]
        graph.add_edge(from_node, to_node, weight=weight)
    
    # Calculate PPR
    return personalized_pagerank(graph, seed_nodes, damping_factor)

