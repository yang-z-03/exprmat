"""
Ligand activity prediction functions
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict
import functools
from .metrics import calculate_auroc, calculate_aupr, classification_evaluation_continuous_pred


def get_single_ligand_importances(
    test_ligand: str,
    ligand_target_matrix: np.ndarray,
    ligand_names: List[str],
    target_names: List[str],
    geneset: List[str],
    background_expressed_genes: List[str],
    known: bool = False
) -> Dict[str, float]:
    """
    Calculate importance scores for a single ligand.
    
    Parameters:
    -----------
    ligand_target_matrix : np.ndarray
        Matrix of ligand-target scores (ligands as columns, targets as rows)
    ligand_names : list of str
        Names of ligands (column names)
    target_names : list of str
        Names of target genes (row names)
    geneset : list of str
        Genes of interest (potentially affected by ligands)
    background_expressed_genes : list of str
        Background genes
    test_ligand : str
        Ligand to test
    known : bool
        Whether this is a known ligand (default False)
    
    Returns:
    --------
    dict
        Dictionary with metrics: test_ligand, auroc, aupr, aupr_corrected, pearson
    """
    # Find ligand index
    if test_ligand not in ligand_names:
        raise ValueError(f"Ligand {test_ligand} not found in ligand_target_matrix")
    
    ligand_idx = ligand_names.index(test_ligand)
    
    # Get predictions for this ligand
    predictions = ligand_target_matrix[:, ligand_idx]
    
    # Create binary labels: 1 for geneset, 0 for background
    labels = np.array([1 if gene in geneset else 0 for gene in target_names])
    
    # Filter to only background + geneset genes
    valid_indices = [i for i, gene in enumerate(target_names) 
                     if gene in geneset or gene in background_expressed_genes]
    
    if len(valid_indices) == 0:
        return {
            'ligand': test_ligand,
            'auroc': 0.5,
            'aupr': 0.0,
            'aupr.adj': 0.0,
            'pearson': 0.0
        }
    
    predictions_filtered = predictions[valid_indices]
    labels_filtered = labels[valid_indices]
    
    # Calculate metrics
    auroc = calculate_auroc(predictions_filtered, labels_filtered)
    aupr = calculate_aupr(predictions_filtered, labels_filtered)
    
    # Calculate Pearson correlation
    if len(predictions_filtered) > 1:
        pearson = np.corrcoef(predictions_filtered, labels_filtered.astype(float))[0, 1]
        if np.isnan(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    
    # AUPR corrected (subtract baseline AUPR)
    # Baseline AUPR = fraction of positive examples
    baseline_aupr = np.mean(labels_filtered)
    aupr_corrected = aupr - baseline_aupr
    
    return {
        'ligand': test_ligand,
        'auroc': auroc,
        'aupr': aupr,
        'aupr.adj': aupr_corrected,
        'pearson': pearson
    }


def predict_ligand_activities(
    geneset: List[str],
    background_expressed_genes: List[str],
    ligand_target_matrix: np.ndarray,
    ligand_names: List[str],
    target_names: List[str],
    potential_ligands: List[str],
    ncpus = 1,
) -> pd.DataFrame:
    """
    Predict activities of ligands in regulating expression of a gene set of interest.
    
    Parameters:
    -----------
    geneset : list of str
        Gene symbols of genes potentially affected by ligands
    background_expressed_genes : list of str
        Background gene symbols
    ligand_target_matrix : np.ndarray
        Ligand-target probability matrix (ligands as columns, targets as rows)
    ligand_names : list of str
        Names of ligands (column names)
    target_names : list of str
        Names of target genes (row names)
    potential_ligands : list of str
        Ligands to test
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: test_ligand, auroc, aupr, aupr_corrected, pearson
    """

    valid_ligands = [ligand for ligand in potential_ligands if ligand in ligand_names]
    from multiprocessing import Pool
    from exprmat import pprog

    with Pool(ncpus) as p:
        results = list(
            pprog(
                p.imap_unordered(
                    functools.partial(
                        get_single_ligand_importances,
                        ligand_target_matrix = ligand_target_matrix,
                        ligand_names = ligand_names,
                        target_names = target_names,
                        geneset = geneset,
                        background_expressed_genes = background_expressed_genes,
                        known = False
                    ),
                    valid_ligands,
                    chunksize = 1,
                ),
                total = len(valid_ligands),
                desc = 'predicting ligand activities'
            )
        )
        
    return pd.DataFrame(results)

