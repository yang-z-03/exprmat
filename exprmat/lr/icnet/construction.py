"""
Network construction functions: weighted networks and hub corrections
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def construct_weighted_networks(
    lr_network: pd.DataFrame,
    sig_network: pd.DataFrame,
    gr_network: pd.DataFrame,
    source_weights_df: pd.DataFrame,
    n_output_networks: int = 2
) -> Dict[str, pd.DataFrame]:
    """
    Construct layer-specific weighted integrated networks from input source networks
    via weighted aggregation.
    
    Parameters:
    -----------
    lr_network : pd.DataFrame
        Ligand-receptor interactions (required columns: from, to, source)
    sig_network : pd.DataFrame
        Signaling interactions (required columns: from, to, source)
    gr_network : pd.DataFrame
        Gene regulatory interactions (required columns: from, to, source)
    source_weights_df : pd.DataFrame
        Weights associated to each data source (required columns: source, weight)
    n_output_networks : int
        Number of output networks: 2 (ligand-signaling and gene regulatory; default)
        or 3 (ligand-receptor, signaling and gene regulatory)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - If n_output_networks=2: {'lr_sig': DataFrame, 'gr': DataFrame}
        - If n_output_networks=3: {'lr': DataFrame, 'sig': DataFrame, 'gr': DataFrame}
        Each DataFrame has columns: from, to, weight
    """
    # Input validation
    if not isinstance(lr_network, pd.DataFrame):
        raise TypeError("lr_network must be a pandas DataFrame")
    if not isinstance(sig_network, pd.DataFrame):
        raise TypeError("sig_network must be a pandas DataFrame")
    if not isinstance(gr_network, pd.DataFrame):
        raise TypeError("gr_network must be a pandas DataFrame")
    if not isinstance(source_weights_df, pd.DataFrame):
        raise TypeError("source_weights_df must be a pandas DataFrame")
    
    required_cols_lr = ['from', 'to', 'source']
    required_cols_weights = ['source', 'weight']
    
    for col in required_cols_lr:
        if col not in lr_network.columns:
            raise ValueError(f"lr_network must have column '{col}'")
        if col not in sig_network.columns:
            raise ValueError(f"sig_network must have column '{col}'")
        if col not in gr_network.columns:
            raise ValueError(f"gr_network must have column '{col}'")
    
    for col in required_cols_weights:
        if col not in source_weights_df.columns:
            raise ValueError(f"source_weights_df must have column '{col}'")
    
    if (source_weights_df['weight'] > 1).any():
        raise ValueError("No data source weight may be higher than 1")
    
    if n_output_networks not in [2, 3]:
        raise ValueError("n_output_networks must be 2 or 3")
    
    # Remove data sources for which weight equals 0
    source_weights_df = source_weights_df[source_weights_df['weight'] > 0].copy()
    
    # Perform weighted network aggregation for gene regulatory network
    gr_network_w = (
        gr_network
        .merge(source_weights_df, on='source', how='inner')
        .groupby(['from', 'to'], as_index=False)
        .agg({'weight': 'sum'})
    )
    
    if n_output_networks == 2:
        # Combine ligand-receptor and signaling networks
        ligand_signaling_w = (
            pd.concat([lr_network, sig_network], ignore_index=True)
            .merge(source_weights_df, on='source', how='inner')
            .groupby(['from', 'to'], as_index=False)
            .agg({'weight': 'sum'})
        )
        return {
            'lr_sig': ligand_signaling_w,
            'gr': gr_network_w
        }
    else:  # n_output_networks == 3
        lr_network_w = (
            lr_network
            .merge(source_weights_df, on='source', how='inner')
            .groupby(['from', 'to'], as_index=False)
            .agg({'weight': 'sum'})
        )
        sig_network_w = (
            sig_network
            .merge(source_weights_df, on='source', how='inner')
            .groupby(['from', 'to'], as_index=False)
            .agg({'weight': 'sum'})
        )
        return {
            'lr': lr_network_w,
            'sig': sig_network_w,
            'gr': gr_network_w
        }


def apply_hub_corrections(
    weighted_networks: Dict[str, pd.DataFrame],
    lr_sig_hub: float,
    gr_hub: float
) -> Dict[str, pd.DataFrame]:
    """
    Apply hub corrections to the weighted integrated ligand-signaling and gene
    regulatory network. Downweighs the importance of nodes with a lot of incoming links.
    
    Hub correction method: W_cor = W * D^(-h)
    where D is the indegree matrix and h is the correction factor.
    
    Parameters:
    -----------
    weighted_networks : dict
        Dictionary with keys 'lr_sig' and 'gr', each containing a DataFrame
        with columns: from, to, weight
    lr_sig_hub : float
        Hub correction factor for ligand-signaling network (0-1).
        0: no correction, 1: maximal correction
    gr_hub : float
        Hub correction factor for gene regulatory network (0-1).
        0: no correction, 1: maximal correction
    
    Returns:
    --------
    dict
        Dictionary with hub-corrected networks: {'lr_sig': DataFrame, 'gr': DataFrame}
    """
    # Input validation
    if not isinstance(weighted_networks, dict):
        raise TypeError("weighted_networks must be a dictionary")
    if 'lr_sig' not in weighted_networks:
        raise ValueError("weighted_networks must contain 'lr_sig' key")
    if 'gr' not in weighted_networks:
        raise ValueError("weighted_networks must contain 'gr' key")
    
    if not isinstance(weighted_networks['lr_sig'], pd.DataFrame):
        raise TypeError("weighted_networks['lr_sig'] must be a pandas DataFrame")
    if not isinstance(weighted_networks['gr'], pd.DataFrame):
        raise TypeError("weighted_networks['gr'] must be a pandas DataFrame")
    
    if 'weight' not in weighted_networks['lr_sig'].columns:
        raise ValueError("lr_sig must contain a column named 'weight'")
    if 'weight' not in weighted_networks['gr'].columns:
        raise ValueError("gr must contain a column named 'weight'")
    
    if not (0 <= lr_sig_hub <= 1):
        raise ValueError("lr_sig_hub must be between 0 and 1 (inclusive)")
    if not (0 <= gr_hub <= 1):
        raise ValueError("gr_hub must be between 0 and 1 (inclusive)")
    
    ligand_signaling_network = weighted_networks['lr_sig'].copy()
    regulatory_network = weighted_networks['gr'].copy()
    
    # Apply hub correction to ligand-signaling network
    if lr_sig_hub > 0:
        # Count incoming edges (indegree) for each node
        indegree = ligand_signaling_network.groupby('to', as_index=False).size()
        indegree.columns = ['to', 'n']
        
        # Merge indegree counts and apply correction
        ligand_signaling_network = (
            ligand_signaling_network
            .merge(indegree, on='to', how='left')
            .assign(weight=lambda x: x['weight'] / (x['n'] ** lr_sig_hub))
            .drop(columns=['n'])
        )
    
    # Apply hub correction to gene regulatory network
    if gr_hub > 0:
        # Count incoming edges (indegree) for each node
        indegree = regulatory_network.groupby('to', as_index=False).size()
        indegree.columns = ['to', 'n']
        
        # Merge indegree counts and apply correction
        regulatory_network = (
            regulatory_network
            .merge(indegree, on='to', how='left')
            .assign(weight=lambda x: x['weight'] / (x['n'] ** gr_hub))
            .drop(columns=['n'])
        )
    
    return {
        'lr_sig': ligand_signaling_network,
        'gr': regulatory_network
    }


def add_new_datasource(
    new_source: pd.DataFrame,
    network: Optional[pd.DataFrame],
    new_weight: float,
    source_weights_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add a new data source to one of the ligand-receptor, signaling or gene
    regulatory data sources.
    
    Parameters:
    -----------
    new_source : pd.DataFrame
        Novel interactions (required columns: from, to, source)
    network : pd.DataFrame or None
        Base network to which to add the new data source (required columns: from, to, source)
        If None, creates a new network
    new_weight : float
        Weight value between 0 and 1 to assign to the new data source
    source_weights_df : pd.DataFrame
        Weights associated to each already included data source (required columns: source, weight)
    
    Returns:
    --------
    tuple
        (updated_network, updated_source_weights_df)
    """
    # Input validation
    if not isinstance(new_source, pd.DataFrame):
        raise TypeError("new_source must be a pandas DataFrame")
    if network is not None and not isinstance(network, pd.DataFrame):
        raise TypeError("network must be a pandas DataFrame or None")
    if not isinstance(source_weights_df, pd.DataFrame):
        raise TypeError("source_weights_df must be a pandas DataFrame")
    
    if not (0 <= new_weight <= 1):
        raise ValueError("new_weight must be between 0 and 1")
    
    if (source_weights_df['weight'] > 1).any() or new_weight > 1:
        raise ValueError("No data source weight may be higher than 1")
    
    # Get unique source name(s) from new_source
    new_source_names = new_source['source'].unique()
    
    if network is None:
        # Create new network
        updated_network = new_source.copy()
        updated_source_weights_df = pd.DataFrame({
            'source': new_source_names,
            'weight': [new_weight] * len(new_source_names)
        })
    else:
        # Add to existing network
        updated_network = pd.concat([network, new_source], ignore_index=True)
        new_weights = pd.DataFrame({
            'source': new_source_names,
            'weight': [new_weight] * len(new_source_names)
        })
        updated_source_weights_df = pd.concat(
            [source_weights_df, new_weights],
            ignore_index=True
        )
    
    return updated_network, updated_source_weights_df

