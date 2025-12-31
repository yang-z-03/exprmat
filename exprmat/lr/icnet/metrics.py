"""
Metrics for evaluating ligand activity predictions
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Tuple, Dict


def calculate_auroc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve (AUROC).
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted scores
    labels : np.ndarray
        True binary labels
    
    Returns:
    --------
    float
        AUROC score
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Cannot calculate ROC with only one class
    return roc_auc_score(labels, predictions)


def calculate_aupr(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Area Under Precision-Recall Curve (AUPR).
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted scores
    labels : np.ndarray
        True binary labels
    
    Returns:
    --------
    float
        AUPR score
    """
    if len(np.unique(labels)) < 2:
        return 0.0  # Cannot calculate PR with only one class
    return average_precision_score(labels, predictions)


def classification_evaluation_continuous_pred(
    predictions: np.ndarray,
    labels: np.ndarray,
    iregulon: bool = False
) -> Dict[str, float]:
    """
    Evaluate classification performance with continuous predictions.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted scores
    labels : np.ndarray
        True binary labels
    iregulon : bool
        Whether to use iRegulon-specific evaluation
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics: auroc, aupr, pearson
    """
    auroc = calculate_auroc(predictions, labels)
    aupr = calculate_aupr(predictions, labels)
    
    # Calculate Pearson correlation
    if len(predictions) > 1:
        pearson = np.corrcoef(predictions, labels.astype(float))[0, 1]
        if np.isnan(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'pearson': pearson
    }

