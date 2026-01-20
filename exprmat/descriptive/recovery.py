
from itertools import repeat
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit

from exprmat.data.signature import signature
from exprmat.data.cistarget import ranking_db
from exprmat.ansi import warning, info, error


def derive_rank_cutoff(
    auc_threshold: float, total_genes: int, rank_threshold: Optional[int] = None
) -> int:

    if not rank_threshold:
        rank_threshold = total_genes - 1

    assert (
        0 < rank_threshold < total_genes
    ), f"rank threshold must be an integer between 1 and {total_genes:d}."
    
    assert (
        0.0 < auc_threshold <= 1.0
    ), "auc threshold must be a fraction between 0.0 and 1.0."

    # in the R implementation the cutoff is rounded.
    rank_cutoff = int(round(auc_threshold * total_genes))
    assert 0 < rank_cutoff <= rank_threshold, (
        f"an auc threshold of {auc_threshold:f} corresponds to {rank_cutoff:d} top "
        f"ranked genes/regions in the database. please increase the rank threshold "
        "or decrease the auc threshold."
    )

    # Make sure we have exactly the same AUC values as the R-SCENIC pipeline.
    # In the latter the rank threshold is not included in AUC calculation.
    rank_cutoff -= 1
    return rank_cutoff


# do not use numba as it dwarfs the performance.
def rcc2d(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int) -> np.ndarray:
    """
    Calculate recovery curves. Each feature is a transcription factor etc, and the top
    rank_threshold genes associated with this factor is taken into consideration.

    Parameters
    ----------
    rankings: 
        The features rankings for a gene signature (n_features, n_genes).
    
    weights: 
        The weights of these genes.
    
    rank_threshold: 
        The total number of ranked genes to take into account when creating a recovery curve.
    
    Returns
    -------
    Recovery curves (n_features, rank_threshold).
    """

    n_features = rankings.shape[0]
    rccs = np.empty(shape = (n_features, rank_threshold))
    for row_idx in range(n_features):
        curranking = rankings[row_idx, :]
        rccs[row_idx, :] = np.cumsum(
            np.bincount(curranking, weights = weights)[:rank_threshold]
        )
    return rccs


def recovery(
    rnk: pd.DataFrame,
    total_genes: int,
    weights: np.ndarray,
    rank_threshold: int,
    auc_threshold: float,
    no_auc: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate recovery curves and AUCs. This is the workhorse of the recovery algorithm.

    Parameters
    ----------
    rnk: 
        A dataframe containing the rank number of genes of interest. 
        Columns correspond to genes.
    
    total_genes: 
        The total number of genes ranked.
    
    weights: 
        The weights associated with the selected genes.
    
    rank_threshold: 
        The total number of ranked genes to take into account when creating a recovery curve.
    
    auc_threshold: 
        The fraction of the ranked genome to take into account for the calculation of the
        area under the recovery curve.
    
    no_auc: 
        Do not calculate AUCs.
    
    Returns
    -------
    A tuple of numpy arrays.
    - The recovery curves (n_features or n_cells * rank_threshold),
    - AUC values (n_features or n_cells).
    """

    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes, rank_threshold)
    features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    weights = np.insert(weights, len(weights), 0.0)
    n_features = len(features)
    rankings = np.append(
        rankings, np.full(shape = (n_features, 1), fill_value = total_genes), axis = 1
    )

    # calculate recovery curves.
    rccs = rcc2d(rankings, weights, rank_threshold)
    if no_auc: return rccs, np.array([])

    maxauc = float((rank_cutoff + 1) * weights.sum())
    assert maxauc > 0
    aucs = rccs[:, :rank_cutoff].sum(axis = 1) / maxauc

    return rccs, aucs


def enrich_cells(
    rnk_mtx: pd.DataFrame, regulon: signature, auc_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Calculate the enrichment of the regulon for the cells in the ranking dataframe.

    Parameters
    ----------
    rnk_mtx: 
        The ranked expression matrix (n_cells, n_genes).
    
    regulon: 
        The regulon the assess for enrichment
    
    auc_threshold: 
        The fraction of the ranked genome to take into account for the calculation of the
        area under the recovery curve.
    """

    total_genes = len(rnk_mtx.columns)
    index = pd.MultiIndex.from_tuples(
        list(zip(rnk_mtx.index.values, repeat(regulon.name))), 
        names = ["cell", "regulon"]
    )

    rnk = rnk_mtx.iloc[:, rnk_mtx.columns.isin(regulon.genes)]
    if rnk.empty or (float(len(rnk.columns)) / float(len(regulon))) < 0.80:
        warning('less than 80% of genes in the regulon is expressed.')
        return pd.DataFrame(index = index, data = {
            "auc": np.zeros(shape = (rnk_mtx.shape[0]), dtype = np.float64)
        })
    
    else:
        weights = np.asarray([
            regulon[gene] if gene in regulon.genes else 1.0
            for gene in rnk.columns.values
        ])
        return pd.DataFrame(index = index, data = {
            "AUC": aucs(rnk, total_genes, weights, auc_threshold)
        })


def enrich_features(
    rnkdb: ranking_db,
    gs: signature,
    rank_threshold: int = 5000,
    auc_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate AUC and NES for all regulatory features in the supplied database using
    the genes of the given signature.
    """
    
    # Load rank of genes from database.
    df = rnkdb.load(gs)
    features = df.index.values
    genes = df.columns.values
    rankings = df.values
    weights = np.asarray([gs[gene] for gene in genes])

    rccs, aucs = recovery(df, rnkdb.n_genes, weights, rank_threshold, auc_threshold)
    ness = (aucs - aucs.mean()) / aucs.std()

    # the creation of a dataframe is a severe performance penalty.
    df_nes = pd.DataFrame(
        index = features, 
        data = {
            ("enrichment", "auc"): aucs, 
            ("enrichment", "nes"): ness
        }
    )

    df_rnks = pd.DataFrame(
        index = features,
        columns = pd.MultiIndex.from_tuples(list(zip(repeat("ranking"), genes))),
        data = rankings,
    )

    df_rccs = pd.DataFrame(
        index = features,
        columns = pd.MultiIndex.from_tuples(
            list(zip(repeat("recovery"), np.arange(rank_threshold)))),
        data = rccs,
    )

    return pd.concat([df_nes, df_rccs, df_rnks], axis = 1)


def leading_edge(
    rcc: np.ndarray,
    avg2stdrcc: np.ndarray,
    ranking: np.ndarray,
    genes: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[str, float]], int]:
    """
    Calculate the leading edge for a given recovery curve.

    Parameters
    ----------
    rcc: 
        The recovery curve.

    avg2stdrcc: 
        The average + 2 standard deviation recovery curve.

    ranking: 
        The rank numbers of the gene signature for a given regulatory feature.

    genes: 
        The genes corresponding to the ranking available in the aforementioned parameter.
    
    weights: 
        The weights for these genes.
    
    Returns
    -------
    The leading edge returned as a list of tuple. Each tuple associates a
    gene part of the leading edge with its rank or with its importance (if gene
    signature supplied). In addition, the rank at maximum difference is returned.
    """

    def critical_point():
        rank_at_max = np.argmax(rcc - avg2stdrcc)
        return rank_at_max, rcc[rank_at_max]

    def get_genes(rank_at_max):
        sorted_idx = np.argsort(ranking)
        sranking = ranking[sorted_idx]
        gene_ids = genes[sorted_idx]
        
        # make sure to include the gene at the leading edge itself. 
        # this is different from the i-cistarget implementation
        # but is inline with the rcistarget implementation.
        filtered_idx = sranking <= rank_at_max
        filtered_gene_ids = gene_ids[filtered_idx]
        return list(
            zip(
                filtered_gene_ids,
                weights[sorted_idx][filtered_idx]
                if weights is not None
                else sranking[filtered_idx],
            )
        )

    rank_at_max, _n_recovered_genes = critical_point()
    return get_genes(rank_at_max), rank_at_max


def leading_edge_row(
    row: pd.Series,
    avg2stdrcc: np.ndarray,
    genes: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Calculate the leading edge for a row of a dataframe. Should be used with partial 
    function application to make this function amenable to the apply idiom common for dataframes.
    """
    return pd.Series(
        data = leading_edge(
            row["recovery"].values, 
            avg2stdrcc, 
            row["ranking"].values, 
            genes, weights
        )
    )


@jit(nopython = True)
def weighted_auc1d(
    ranking: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUC of the weighted recovery curve of a single ranking.
    """
    # using concatenate and full constructs required by numba.
    # The rankings are 0-based. The position at the rank threshold is included 
    # in the calculation.
    filter_idx = ranking < rank_cutoff
    x = ranking[filter_idx]
    y = weights[filter_idx]
    sort_idx = np.argsort(x)
    x = np.concatenate((x[sort_idx], np.full((1,), rank_cutoff, dtype=np.int_)))
    y = y[sort_idx].cumsum()
    return np.sum(np.diff(x) * y) / max_auc


def auc2d(
    rankings: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUCs of multiple rankings.
    """
    n_features = rankings.shape[0]
    aucs = np.empty(shape=(n_features,), dtype = np.float64)  # Pre-allocation.
    for row_idx in range(n_features):
        aucs[row_idx] = weighted_auc1d(
            rankings[row_idx, :], weights, rank_cutoff, max_auc)
    return aucs


def aucs(
    rnk: pd.DataFrame, total_genes: int, weights: np.ndarray, auc_threshold: float
) -> np.ndarray:
    """
    Calculate AUCs (implementation without calculating recovery curves first).

    Parameters
    ----------
    rnk: 
        A dataframe containing the rank number of genes of interest. 
        Columns correspond to genes.

    total_genes: 
        The total number of genes ranked.

    weights: 
        The weights associated with the selected genes.

    auc_threshold: 
        The fraction of the ranked genome to take into account for the calculation of the
        area under the recovery curve.
    """
    
    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes)
    _features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    y_max = weights.sum()
    
    # the rankings are 0-based. the position at the rank threshold is included in the calculation.
    # the maximum auc takes this into account.
    # for reason of generating the same results as in r we introduce an error by adding 
    # one to the rank_cutoff for calculationg the maximum auc.
    maxauc = float((rank_cutoff + 1) * y_max)
    assert maxauc > 0
    return auc2d(rankings, weights, rank_cutoff, maxauc)
