"""
Ligand-target matrix construction using Personalized PageRank
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Optional, Union
import networkx as nx

from .pagerank import personalized_pagerank


def construct_ligand_tf_matrix(
    weighted_networks: Dict[str, pd.DataFrame],
    lr_network: pd.DataFrame,
    ligands: List[Union[str, List[str]]],
    ltf_cutoff: float = 0.99,
    algorithm: str = "PPR",
    damping_factor: float = 0.5,
    ligands_as_cols: bool = False
) -> np.ndarray:
    """
    Construct a ligand-TF signaling probability matrix.
    
    Parameters:
    -----------
    weighted_networks : dict
        Dictionary with 'lr_sig' and 'gr' DataFrames (columns: from, to, weight)
    lr_network : pd.DataFrame
        Ligand-receptor network (columns: from, to, source)
    ligands : list
        List of ligands or ligand combinations. Example: ["TNF", "BMP2", ["IL4", "IL13"]]
    ltf_cutoff : float
        Quantile cutoff. Scores below this quantile will be set to 0 (default 0.99)
    algorithm : str
        Algorithm to use: "PPR" (default), "SPL", or "direct"
    damping_factor : float
        Damping factor for PPR algorithm (default 0.5)
    ligands_as_cols : bool
        If True, ligands are columns and TFs are rows. If False, vice versa.
    
    Returns:
    --------
    np.ndarray
        Ligand-TF matrix with probability scores
    """
    # Input validation
    if algorithm not in ["PPR", "SPL", "direct"]:
        raise ValueError("algorithm must be 'PPR', 'SPL', or 'direct'")
    
    if not (0 <= ltf_cutoff <= 1):
        raise ValueError("ltf_cutoff must be between 0 and 1")
    
    if algorithm == "PPR" and not (0 <= damping_factor < 1):
        raise ValueError("damping_factor must be between 0 and 1 (1 not included)")
    
    # Get networks
    ligand_signaling_network = weighted_networks['lr_sig'].copy()
    regulatory_network = weighted_networks['gr'].copy()
    
    # Get all unique genes
    all_genes = sorted(set(
        list(ligand_signaling_network['from'].unique()) +
        list(ligand_signaling_network['to'].unique()) +
        list(regulatory_network['from'].unique()) +
        list(regulatory_network['to'].unique())
    ))
    
    # Create gene to index mapping
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    idx_to_gene = {i: gene for gene, i in gene_to_idx.items()}
    
    # Convert network edges to integer indices
    ligand_signaling_edges = ligand_signaling_network.copy()
    ligand_signaling_edges['from_idx'] = ligand_signaling_edges['from'].map(gene_to_idx)
    ligand_signaling_edges['to_idx'] = ligand_signaling_edges['to'].map(gene_to_idx)
    ligand_signaling_edges = ligand_signaling_edges[['from_idx', 'to_idx', 'weight']].dropna()
    
    # Build graph for PPR
    graph = nx.DiGraph()
    for _, row in ligand_signaling_edges.iterrows():
        from_gene = idx_to_gene[int(row['from_idx'])]
        to_gene = idx_to_gene[int(row['to_idx'])]
        graph.add_edge(from_gene, to_gene, weight=row['weight'])
    
    # Process each ligand/ligand combination
    ligand_tf_scores = []
    ligand_names = []
    
    for ligand_set in ligands:
        if isinstance(ligand_set, str):
            ligand_set = [ligand_set]
        
        ligand_name = "-".join(ligand_set)
        ligand_names.append(ligand_name)
        
        if algorithm == "PPR":
            # Calculate PPR for this ligand set
            ppr_scores = personalized_pagerank(
                graph, 
                seed_nodes=ligand_set,
                damping_factor=damping_factor
            )
            
            # Convert to array indexed by gene
            scores = np.zeros(len(all_genes))
            for i, gene in enumerate(all_genes):
                if gene in ppr_scores:
                    scores[i] = ppr_scores[gene]
            
        elif algorithm == "SPL":
            # Shortest path length algorithm
            # Reverse weights (1/weight) for shortest path
            graph_spl = graph.copy()
            for u, v, data in graph_spl.edges(data=True):
                if data['weight'] > 0:
                    graph_spl[u][v]['weight'] = 1.0 / data['weight']
            
            scores = np.zeros(len(all_genes))
            for ligand in ligand_set:
                if ligand in graph_spl:
                    # Calculate shortest paths from ligand to all nodes
                    try:
                        lengths = nx.single_source_dijkstra_path_length(
                            graph_spl, ligand, weight='weight'
                        )
                        for gene, length in lengths.items():
                            if gene in gene_to_idx:
                                idx = gene_to_idx[gene]
                                # Convert path length back to score (inverse)
                                scores[idx] = max(scores[idx], 1.0 / (1.0 + length))
                    except nx.NetworkXNoPath:
                        pass
            
        else:  # algorithm == "direct"
            # Direct weights from ligand-signaling network
            scores = np.zeros(len(all_genes))
            for ligand in ligand_set:
                if ligand in gene_to_idx:
                    ligand_idx = gene_to_idx[ligand]
                    # Get direct connections from this ligand
                    ligand_edges = ligand_signaling_edges[
                        ligand_signaling_edges['from_idx'] == ligand_idx
                    ]
                    if len(ligand_edges) > 0:
                        max_weight = ligand_edges['weight'].max()
                        for _, row in ligand_edges.iterrows():
                            to_idx = int(row['to_idx'])
                            scores[to_idx] = max(scores[to_idx], row['weight'] / max_weight)
            
            # Give ligand itself max score
            for ligand in ligand_set:
                if ligand in gene_to_idx:
                    scores[gene_to_idx[ligand]] = 1.0
        
        # Apply cutoff
        if ltf_cutoff > 0:
            cutoff_value = np.quantile(scores[scores > 0], ltf_cutoff) if np.any(scores > 0) else 0
            scores[scores < cutoff_value] = 0
        
        ligand_tf_scores.append(scores)
    
    # Create matrix
    ltf_matrix = np.array(ligand_tf_scores)
    
    # Set row and column names
    if ligands_as_cols:
        ltf_matrix = ltf_matrix.T
    
    return ltf_matrix, ligand_names, all_genes


def construct_tf_target_matrix(
    weighted_networks: Dict[str, pd.DataFrame],
    tfs_as_cols: bool = False,
    standalone_output: bool = False
) -> np.ndarray:
    """
    Construct a TF-target matrix from the gene regulatory network.
    
    Parameters:
    -----------
    weighted_networks : dict
        Dictionary with 'lr_sig' and 'gr' DataFrames
    tfs_as_cols : bool
        If True, TFs are columns and targets are rows
    standalone_output : bool
        If True, only include regulators with gene regulatory interactions
    
    Returns:
    --------
    np.ndarray
        TF-target matrix
    """
    regulatory_network = weighted_networks['gr'].copy()
    ligand_signaling_network = weighted_networks['lr_sig'].copy()
    
    # Get all genes
    all_genes = sorted(set(
        list(ligand_signaling_network['from'].unique()) +
        list(ligand_signaling_network['to'].unique()) +
        list(regulatory_network['from'].unique()) +
        list(regulatory_network['to'].unique())
    ))
    
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    
    # Build sparse matrix
    n = len(all_genes)
    rows = []
    cols = []
    data = []
    
    for _, row in regulatory_network.iterrows():
        from_idx = gene_to_idx.get(row['from'])
        to_idx = gene_to_idx.get(row['to'])
        if from_idx is not None and to_idx is not None:
            rows.append(from_idx)
            cols.append(to_idx)
            data.append(row['weight'])
    
    grn_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    grn_matrix = grn_matrix.toarray()
    
    # Set row/column names
    if standalone_output:
        # Keep only regulators
        regulators = set(regulatory_network['from'].unique())
        regulator_indices = [i for i, gene in enumerate(all_genes) if gene in regulators]
        grn_matrix = grn_matrix[regulator_indices, :]
    
    if tfs_as_cols:
        grn_matrix = grn_matrix.T
    
    return grn_matrix, all_genes


def construct_ligand_target_matrix(
    weighted_networks: Dict[str, pd.DataFrame],
    lr_network: pd.DataFrame,
    ligands: List[Union[str, List[str]]],
    ltf_cutoff: float = 0.99,
    algorithm: str = "PPR",
    damping_factor: float = 0.5,
    secondary_targets: bool = False,
    ligands_as_cols: bool = True,
    remove_direct_links: str = "no"
) -> np.ndarray:
    """
    Construct a ligand-target probability matrix.
    
    Parameters:
    -----------
    weighted_networks : dict
        Dictionary with 'lr_sig' and 'gr' DataFrames
    lr_network : pd.DataFrame
        Ligand-receptor network
    ligands : list
        List of ligands or ligand combinations
    ltf_cutoff : float
        Quantile cutoff for ligand-TF scores
    algorithm : str
        Algorithm: "PPR" (default), "SPL", or "direct"
    damping_factor : float
        Damping factor for PPR
    secondary_targets : bool
        Include secondary targets (default False)
    ligands_as_cols : bool
        Ligands as columns (default True)
    remove_direct_links : str
        "no", "ligand", or "ligand-receptor"
    
    Returns:
    --------
    np.ndarray
        Ligand-target matrix
    """
    # Remove direct links if required
    if remove_direct_links != "no":
        ligands_set = set(lr_network['from'].unique())
        receptors_set = set(lr_network['to'].unique())
        
        if remove_direct_links == "ligand":
            weighted_networks['gr'] = weighted_networks['gr'][
                ~weighted_networks['gr']['from'].isin(ligands_set)
            ].copy()
        elif remove_direct_links == "ligand-receptor":
            weighted_networks['gr'] = weighted_networks['gr'][
                ~weighted_networks['gr']['from'].isin(ligands_set | receptors_set)
            ].copy()
    
    # Construct ligand-TF matrix
    ltf_matrix, ligand_names, all_genes = construct_ligand_tf_matrix(
        weighted_networks, lr_network, ligands, ltf_cutoff, 
        algorithm, damping_factor, ligands_as_cols=False
    )
    
    # Construct TF-target matrix
    grn_matrix, all_genes_gr = construct_tf_target_matrix(
        weighted_networks, tfs_as_cols=False, standalone_output=False
    )
    
    # Ensure gene lists match
    if len(all_genes) != len(all_genes_gr) or all_genes != all_genes_gr:
        # Align matrices
        common_genes = sorted(set(all_genes) & set(all_genes_gr))
        # Reindex matrices to common genes
        # (simplified - in practice, need proper alignment)
        pass
    
    # Multiply ligand-TF matrix with TF-target matrix
    ligand_to_target = ltf_matrix @ grn_matrix
    
    # Secondary targets
    if secondary_targets:
        ltf_matrix_secondary = ligand_to_target.copy()
        
        if ltf_cutoff > 0:
            for i in range(ltf_matrix_secondary.shape[0]):
                row = ltf_matrix_secondary[i, :]
                cutoff_value = np.quantile(row[row > 0], ltf_cutoff) if np.any(row > 0) else 0
                ltf_matrix_secondary[i, row < cutoff_value] = 0
        
        ligand_to_target_secondary = ltf_matrix_secondary @ grn_matrix
        ligand_to_target = ligand_to_target + ligand_to_target_secondary
    
    # Transpose if needed
    if ligands_as_cols:
        ligand_to_target = ligand_to_target.T
    
    return ligand_to_target, ligand_names, all_genes

