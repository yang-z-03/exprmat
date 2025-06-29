
import pandas as pd
from functools import partial
from itertools import chain
from typing import Sequence

import numpy as np
import pandas as pd
from numba import float64, int64, njit, prange

from exprmat.data.signature import regulon, signature
from exprmat.ansi import error, info, warning
from exprmat.data.io import zopen

COLUMN_NAME_TF = "tf"
COLUMN_NAME_MOTIF_ID = "motif"
COLUMN_NAME_MOTIF_SIMILARITY_QVALUE = "similarity_q"
COLUMN_NAME_ORTHOLOGOUS_IDENTITY = "orthologous"
COLUMN_NAME_ANNOTATION = "annotation"

ACTIVATING_MODULE = "activating"
REPRESSING_MODULE = "repressing"

COLUMN_NAME_TARGET = "target"
COLUMN_NAME_WEIGHT = "importance"
COLUMN_NAME_REGULATION = "regulation"
COLUMN_NAME_CORRELATION = "rho"
RHO_THRESHOLD = 0.03


def module_threshold(
    adjacencies, threshold, 
    context = frozenset(), pattern = "weight > {:.3f}"
):

    for tf_name, df_grp in adjacencies[
        adjacencies[COLUMN_NAME_WEIGHT] > threshold
    ].groupby(by = COLUMN_NAME_TF):
        if len(df_grp) > 0:
            yield regulon(
                name = "Regulon for {}".format(tf_name),
                context = frozenset([pattern.format(threshold)]).union(context),
                transcription_factor = tf_name,
                gene2occurrence = [],
                gene2weight = list(zip(
                    df_grp[COLUMN_NAME_TARGET].values,
                    df_grp[COLUMN_NAME_WEIGHT].values,
                )),
            )


def module_top_targets(
    adjacencies, n, context = frozenset()
):

    for tf_name, df_grp in adjacencies.groupby(by = COLUMN_NAME_TF):
        module = df_grp.nlargest(n, COLUMN_NAME_WEIGHT)
        if len(module) > 0:
            yield regulon(
                name = "Regulon for {}".format(tf_name),
                context = frozenset(["top({})".format(n)]).union(context),
                transcription_factor = tf_name,
                gene2occurrence = [],
                gene2weight = list(zip(
                    module[COLUMN_NAME_TARGET].values,
                    module[COLUMN_NAME_WEIGHT].values,
                )),
            )


def module_top_factors(
    adjacencies, n, context = frozenset()
):

    df = adjacencies.groupby(by = adjacencies[COLUMN_NAME_TARGET].rename(None)).apply(
        lambda grp: grp.nlargest(n, COLUMN_NAME_WEIGHT), include_groups = False)
    for tf_name, df_grp in df.groupby(by = COLUMN_NAME_TF):
        if len(df_grp) > 0:
            yield regulon(
                name = tf_name,
                context = frozenset(["toptarget({})".format(n)]).union(context),
                transcription_factor = tf_name,
                gene2occurrence = [],
                gene2weight = list(zip(
                    df_grp[COLUMN_NAME_TARGET].values,
                    df_grp[COLUMN_NAME_WEIGHT].values,
                )),
            )


def modules_from_adjacencies(
    adjacencies: pd.DataFrame,
    ex_mtx: pd.DataFrame,
    thresholds = (0.75, 0.90),
    top_n_targets = (50,),
    top_n_regulators = (5, 10, 50),
    min_genes = 20,
    absolute_thresholds = False,
    rho_dichotomize = True,
    keep_only_activating = True,
    rho_threshold = RHO_THRESHOLD,
    rho_mask_dropouts = False,
) -> Sequence[regulon]:
    """
    Create modules from a dataframe containing weighted adjacencies between a 
    TF and its target genes.

    Parameters
    ----------
    adjacencies: 
        The dataframe with the TF-target links. This dataframe should have the following columns:
        ['tf', 'target', 'weight'].

    ex_mtx: 
        The expression matrix (n_cells * n_genes).
        Expression matrices are given in pandas dataframe.

    thresholds: 
        The first method to create the TF-modules based on the best targets for each 
        transcription factor.

    top_n_targets: 
        The second method is to select the top targets for a given TF.
    
    top_n_regulators: 
        The alternative way to create the TF-modules is to select the best 
        regulators for each gene.

    min_genes: 
        The required minimum number of genes in a resulting module.

    absolute_thresholds: 
        Use absolute thresholds or percentiles to define modules based on 
        best targets of a TF.

    rho_dichotomize: 
        Differentiate between activating and repressing modules based on the 
        correlation patterns of the expression of the TF and its target genes.

    keep_only_activating: 
        Keep only modules in which a TF activates its target genes.

    rho_threshold: 
        The threshold on the correlation to decide if a target gene is activated
        (rho > `rho_threshold`) or repressed (rho < -`rho_threshold`).

    rho_mask_dropouts: 
        Do not use cells in which either the expression of the TF or the target gene 
        is 0 when calculating the correlation between a TF-target pair.

    Returns a sequence of regulons.
    """

    # duplicate genes need to be removed from the expression matrix to avoid lookup 
    # problems in the correlation matrix. in addition, also make sure the expression 
    # matrix consists of floating point numbers. this requirement might
    # be violated when dealing with raw counts as input.

    ex_mtx = ex_mtx.T[~ ex_mtx.columns.duplicated(keep = "first")].T.astype(float)

    # to make the pyscenic code more robust to the selection of the network inference 
    # method in the first step of the pipeline, it is better to use percentiles instead 
    # of absolute values for the weight thresholds.

    if not absolute_thresholds:

        def iter_modules(adjc, context):
            yield from chain(
                chain.from_iterable(
                    module_threshold(
                        adjc, thr, context, pattern = "weight > {} pct.".format(frac * 100))
                    for thr, frac in zip(
                        list(adjacencies[COLUMN_NAME_WEIGHT].quantile(thresholds)),
                        thresholds
                    )
                ),
                chain.from_iterable(
                    module_top_targets(adjc, n, context) for n in top_n_targets),
                chain.from_iterable(
                    module_top_factors(adjc, n, context) for n in top_n_regulators),
            )

    else:

        def iter_modules(adjc, context):
            yield from chain(
                chain.from_iterable(
                    module_threshold(adjc, thr, context) for thr in thresholds),
                chain.from_iterable(
                    module_top_targets(adjc, n, context) for n in top_n_targets),
                chain.from_iterable(
                    module_top_factors(adjc, n, context) for n in top_n_regulators),
            )

    if not rho_dichotomize:
        # do not differentiate between activating and repressing modules.
        modules_iter = iter_modules(adjacencies, frozenset())
    
    else:

        # relationship between tf and its target, i.e. activator or repressor, 
        # is derived using the original expression profiles. The pearson product-moment 
        # correlation coefficient is used to derive this information.

        if not {"regulation", "rho"}.issubset(adjacencies.columns):
            # Add correlation column and create two disjoint set of adjacencies.
            info("calculating pearson correlations ...")
            # test for genes present in the adjacencies but not present in the expression matrix:
            unique_adj_genes = set(adjacencies[COLUMN_NAME_TF]).union(
                set(adjacencies[COLUMN_NAME_TARGET])
            ) - set(ex_mtx.columns)
            if len(unique_adj_genes) != 0:
                warning(f"found {len(unique_adj_genes)} genes present in the network (adjacencies) output")
                warning(f"but missing from the expression matrix.")
            
            adjacencies = add_correlation(
                adjacencies,
                ex_mtx,
                rho_threshold = rho_threshold,
                mask_dropouts = rho_mask_dropouts,
            )

        else: info("use existing pearson correlations from the adjacencies table.")

        activating_modules = adjacencies[adjacencies[COLUMN_NAME_REGULATION] > 0.0]
        if keep_only_activating:
            modules_iter = iter_modules(
                activating_modules, frozenset([ACTIVATING_MODULE]))
            
        else:
            repressing_modules = adjacencies[adjacencies[COLUMN_NAME_REGULATION] < 0.0]
            modules_iter = chain(
                iter_modules(activating_modules, frozenset([ACTIVATING_MODULE])),
                iter_modules(repressing_modules, frozenset([REPRESSING_MODULE])),
            )

    # derive modules for these adjacencies.
    #
    # - add the transcription factor to the module. we are unable to assess if a tf works 
    #   in a direct self-regulating way, either inhibiting its own expression or activating 
    #   it. therefore the most unbiased way forward is to add the tf to both activating as 
    #   well as repressing modules
    #
    # - filter for minimum number of genes.

    def add_tf(module): return module.add(module.transcription_factor)
    return list(filter(lambda m: len(m) >= min_genes, map(add_tf, modules_iter)))


def create_id_pairs(adjacencies: pd.DataFrame, exp_mtx: pd.DataFrame) -> np.ndarray:

    # create sorted list of genes that take part in a tf-target link.
    genes = set(adjacencies.tf).union(set(adjacencies.target))
    sorted_genes = sorted(genes)

    # find column idx in the expression matrix of each gene that takes part in a link. 
    # having the column index of genes sorted as well as the list of link genes makes 
    # sure that we can map indexes back to genes! this only works if all genes we are 
    # looking for are part of the expression matrix.
    assert len(set(exp_mtx.columns).intersection(genes)) == len(genes)
    symbol2idx = dict(
        zip(sorted_genes, np.nonzero(exp_mtx.columns.isin(sorted_genes))[0])
    )

    # create numpy array of idx pairs.
    return np.array([
        [symbol2idx[s1], symbol2idx[s2]]
        for s1, s2 in zip(adjacencies.tf, adjacencies.target)
    ])


@njit(float64(float64[:], float64[:], float64))
def masked_rho(x: np.ndarray, y: np.ndarray, mask: float = 0.0) -> float:

    idx = (x != mask) & (y != mask)
    x_masked = x[idx]
    y_masked = y[idx]
    if (len(x_masked) == 0) or (len(y_masked) == 0):
        return np.nan
    x_demeaned = x_masked - x_masked.mean()
    y_demeaned = y_masked - y_masked.mean()
    cov_xy = np.dot(x_demeaned, y_demeaned)
    std_x = np.sqrt(np.dot(x_demeaned, x_demeaned))
    std_y = np.sqrt(np.dot(y_demeaned, y_demeaned))
    if (std_x * std_y) == 0:
        return np.nan
    return cov_xy / (std_x * std_y)


@njit(float64[:](float64[:, :], int64[:, :], float64), parallel=True)
def masked_rho_pairs(
    mtx: np.ndarray, col_idx_pairs: np.ndarray, mask: float = 0.0
) -> np.ndarray:
    """
    Calculates the masked correlation of columns pairs in a matrix.

    - mtx: the matrix from which columns will be used.
    - col_idx_pairs: the pairs of column indexes (n * 2).
    
    Returns an array with correlation coefficients (n).
    """
    # numba can parallelize loops automatically 
    # but this is still an experimental feature.

    n = col_idx_pairs.shape[0]
    rhos = np.empty(shape = n, dtype = np.float64)
    for n_idx in prange(n):
        x = mtx[:, col_idx_pairs[n_idx, 0]]
        y = mtx[:, col_idx_pairs[n_idx, 1]]
        rhos[n_idx] = masked_rho(x, y, mask)
    return rhos


def add_correlation(
    adjacencies: pd.DataFrame,
    ex_mtx: pd.DataFrame,
    rho_threshold = RHO_THRESHOLD,
    mask_dropouts = False,
) -> pd.DataFrame:
    
    assert rho_threshold > 0, "rho_threshold should be greater than 0."

    # TODO: use spearman correlation instead of pearson correlation coefficient: 
    # using a non-parametric test like spearman rank correlation makes much more 
    # sense because we want to capture monotonic and not specifically linear
    # relationships between tf and target genes.

    # assessment of best optimization strategy for calculating dropout masked 
    # correlations between tf-target expression:
    #
    # measurement of time performance of masked_rho (with numba jit): 136 µs ± 932 ns 
    # for a single pair of vectors. for a typical dataset this translates into 
    # (for a single core):
    # 
    # 1. calculating the rectangular (tf * target) correlation matrix:
    #    (1,564 tfs * 19,812 targets * 136 ms * 10e-6) / 3600.0 ~ 12 hours.
    #    this approach calculates far too much be has the potential for easy 
    #    parallelization via numba (cf. current implementation of masked_rho_2d).
    #
    # 2. calculating only needed tf-target pairs:
    #    (6,732,441 tf-target links * 136 ms * 10e-6)/3600.0 ~ 2h 30 mins.
    #    
    #    many of these gene-gene links will be duplicate so there might be a potential 
    #    for memoization. however because the calculation is already quite fast and 
    #    the memoization would need to take into account the commutativity of the operation 
    #    and involves hashing large numerical vectors, the benefit if this memoization 
    #    might be minimal.
    #    
    #    calculation of unique pairs already takes substantial amount of time and 
    #    does not introduce a substantial reduction in the number of gene-gene pairs to 
    #    calculate the correlation for: 6,732,441 => 6,630,720 (2 min 9 s).
    #    this is exactly the additional needed for calculating the rho values for these 
    #    pairs. no gain here.
    #
    # the other options would have been to used the masked array abstraction provided 
    # by numpy but this again this not allow for easy parallelization. in addition the 
    # corrcoef operation is far slower than the numba jit implementation: 2.36 ms ± 62 µs 
    # per loop. the best combined approach is to calculate rhos for pairs defined by 
    # indexes which is the approach implemented below.

    # calculate pearson correlation to infer repression or activation.
    if mask_dropouts:
        ex_mtx = ex_mtx.sort_index(axis = 1)
        col_idx_pairs = create_id_pairs(adjacencies, ex_mtx)
        rhos = masked_rho_pairs(ex_mtx.values, col_idx_pairs, 0.0)
    else:
        genes = list(
            set(adjacencies[COLUMN_NAME_TF])
            .union(set(adjacencies[COLUMN_NAME_TARGET]))
        )

        ex_mtx = ex_mtx[ex_mtx.columns[ex_mtx.columns.isin(genes)]]
        corr_mtx = pd.DataFrame(
            index = ex_mtx.columns,
            columns = ex_mtx.columns,
            data = np.corrcoef(ex_mtx.values.T),
        )

        rhos = np.array(
            [corr_mtx[s2][s1] for s1, s2 in zip(adjacencies.TF, adjacencies.target)])

    regulations = (rhos > rho_threshold).astype(int) - (rhos < -rho_threshold).astype(int)

    return pd.DataFrame(
        data={
            COLUMN_NAME_TF: adjacencies[COLUMN_NAME_TF].values,
            COLUMN_NAME_TARGET: adjacencies[COLUMN_NAME_TARGET].values,
            COLUMN_NAME_WEIGHT: adjacencies[COLUMN_NAME_WEIGHT].values,
            COLUMN_NAME_REGULATION: regulations,
            COLUMN_NAME_CORRELATION: rhos,
        }
    )
