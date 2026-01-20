
import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import rankdata, beta


def aggregate(
    lrs: dict,
    consensus,
    aggregate_method: str = 'rra',
    consensus_opts: list = None,
    key_cols: list = None
) -> pd.DataFrame:
    """
    Parameters
    ---------
    lrs
        A list with results for all methods
    
    consensus
        Instance used to generate the lr results
    
    key_cols
        Should represent unique LRs columns by which to join
    
    aggregate_method
        Method by which we aggregate the ranks. Options are ['rra', 'mean'], where 'rra' 
        corresponds to the RRA method, while 'mean' is just the mean of the ranks divided 
        by the number of interactions
    
    consensus_opts
        consensus ranks to be obtained

    Returns
    -------
    A long pd.DataFrame with ranked LRs
    """

    # join the sc to the whole universe between the methods
    if key_cols is None:
        key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
    if consensus_opts is None:
        consensus_opts = ['Magnitude', 'Specificity']

    lrs = [lrs[method].drop_duplicates(keep='first') for method in lrs]
    
    # reduce to a df with the shared keys + all relevant sc
    lr_res = reduce(
        lambda left, right:
        pd.merge(left, right, how = 'outer', on = key_cols,suffixes = ('', '_duplicated')), lrs
    )
    
    # drop duplicated columns
    lr_res = lr_res.loc[:, ~lr_res.columns.str.endswith('_duplicated')]

    order_col = ''
    if 'Specificity' in consensus_opts:
        lr_res[consensus.specificity] = rank_aggregate(
            lr_res.copy(), consensus.specificity_specs, aggregate_method = aggregate_method)
        order_col = consensus.specificity
    
    if 'Magnitude' in consensus_opts:
        lr_res[consensus.magnitude] = rank_aggregate(
            lr_res.copy(), consensus.magnitude_specs, aggregate_method = aggregate_method)
        order_col = consensus.magnitude

    lr_res = lr_res.sort_values(order_col)
    return lr_res


def rank_aggregate(lr_res, specs, aggregate_method) -> np.array:

    assert aggregate_method in ['rra', 'mean']

    # convert specs columns to ranks
    for spec in specs:
        score_name = specs[spec][0]
        ascending = specs[spec][1]

        if ascending: lr_res[score_name] = rankdata(lr_res[score_name], method = 'average')
        else: lr_res[score_name] = rankdata(lr_res[score_name] * -1, method = 'average')

    # get only the relevant ranks as a mat (joins order the keys)
    scores = list({specs[s][0] for s in specs})
    rmat = lr_res[scores].values

    if aggregate_method == 'rra':
        return robust_rank_aggregate(rmat)
    elif aggregate_method == 'mean':
        return np.mean(rmat, axis=1) / rmat.shape[0]


def correct_beta_pvals(p, k) -> np.array:
    p = np.clip(p * k, a_min = 0, a_max = 1)
    return p


def rho_score(rmat, dist_a, dist_b):
    """ Calculate Beta Distribution Rho Scores """

    # Sort values by sources (rows)
    rmat = np.sort(rmat, axis=1)
    # Calc beta cdf across rows
    p = beta.cdf(rmat, dist_a, dist_b)
    # get min pval per row
    p = np.min(p, axis=1)
    # correct p-vals
    rho = correct_beta_pvals(p, k = rmat.shape[1])

    return rho


def robust_rank_aggregate(rmat) -> np.array:

    # 0-1 values depending on relative rank of each interaction divided by the 
    # max of each method due to max diffs due to ties
    rmat = rmat / np.max(rmat, axis = 0)

    # generate dist_a / b with same row size as rmat
    dist_a = np.repeat([np.arange(rmat.shape[1])], rmat.shape[0], axis = 0) + 1
    dist_b = rmat.shape[1] - dist_a + 1
    return rho_score(rmat, dist_a, dist_b)
