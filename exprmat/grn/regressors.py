
import numpy as np
import pandas as pd

import scipy
import scipy.sparse
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor, 
    ExtraTreesRegressor
)

from dask.dataframe.utils import make_meta
from exprmat.ansi import error, info, warning

DEMON_SEED = 666
ANGEL_SEED = 777
EARLY_STOP_WINDOW_LENGTH = 25


sklearn_regressors = {
    'rf': RandomForestRegressor,
    'et': ExtraTreesRegressor,
    'gbm': GradientBoostingRegressor
}

kwargs_random_forest = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn extra-trees regressor
kwargs_extra_trees = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

# scikit-learn gradient boosting regressor
kwargs_gbm = {
    'learning_rate': 0.01,
    'n_estimators': 500,
    'max_features': 0.1
}

# scikit-learn stochastic gradient boosting regressor
kwargs_stochastic_gbm = {
    'learning_rate': 0.01,
    'n_estimators': 5000,  # can be arbitrarily large
    'max_features': 0.1,
    'subsample': 0.9
}

grn_schema = make_meta({'tf': str, 'target': str, 'importance': float})
meta_schema = make_meta({'target': str, 'n_estimators': int})


def is_sklearn_regressor(regressor_type):
    return regressor_type.lower() in sklearn_regressors.keys()


def is_xgboost_regressor(regressor_type):
    return regressor_type.lower() == 'xgb'


def is_oob_heuristic_supported(regressor_type, regressor_kwargs):
    return (
        regressor_type.lower() == 'gbm' and 
        'subsample' in regressor_kwargs and 
        regressor_kwargs['subsample'] < 1.0
    )


def to_tf_matrix(expression_matrix, gene_names, tf_names):

    tuples = [(index, gene) for index, gene in enumerate(gene_names) if gene in tf_names]
    tf_indices = [t[0] for t in tuples]
    tf_matrix_names = [t[1] for t in tuples]
    return expression_matrix[:, tf_indices], tf_matrix_names


def fit_model(
    regressor_type,
    regressor_kwargs,
    tf_matrix,
    target_gene_expression,
    early_stop_window_length = EARLY_STOP_WINDOW_LENGTH,
    seed = DEMON_SEED
):
    """
    Parameters
    ----------
    regressor_type: string
        Case insensitive. Regressor types in 'gbm', 'rf', 'et'.
    
    regressor_kwargs: dict
        A dictionary of key-value pairs that configures the regressor.
    
    tf_matrix
        The predictor matrix (transcription factor matrix) as a numpy array.
        Rows are cells and columns are transcriptional factors

    target_gene_expression
        The target (y) gene expression to predict in function of the tf_matrix (X).
    
    early_stop_window_length
        Window length of the early stopping monitor.

    seed
        (optional) Random seed for the regressors.
        
    Returns
    -------
    A trained regression model.
    """

    regressor_type = regressor_type.lower()
    if isinstance(target_gene_expression, scipy.sparse.spmatrix):
        target_gene_expression = target_gene_expression.todense().A.flatten()
    assert tf_matrix.shape[0] == target_gene_expression.shape[0]

    def do_sklearn_regression():
        regressor = sklearn_regressors[regressor_type](random_state = seed, **regressor_kwargs)
        with_early_stopping = is_oob_heuristic_supported(regressor_type, regressor_kwargs)

        if with_early_stopping:
            regressor.fit(
                tf_matrix, target_gene_expression, 
                monitor = early_stop_monitor(early_stop_window_length)
            )

        else: regressor.fit(tf_matrix, target_gene_expression)
        return regressor

    if is_sklearn_regressor(regressor_type):
        return do_sklearn_regression()
    else: error('unsupported regressor type: {0}'.format(regressor_type))


def to_feature_importances(
    regressor_type, regressor_kwargs, trained_regressor
):
    """
    When the out-of-bag improvement heuristic is used, we cancel the effect 
    of normalization by dividing by the number of trees in the regression 
    ensemble by multiplying again by the number of trees used.

    Returns
    -------
    The feature importances inferred from the trained model.
    """

    if is_oob_heuristic_supported(regressor_type, regressor_kwargs):
        n_estimators = len(trained_regressor.estimators_)
        denormalized_importances = trained_regressor.feature_importances_ * n_estimators
        return denormalized_importances
    else: return trained_regressor.feature_importances_


def to_meta_df(trained_regressor, target_gene_name):
    n_estimators = len(trained_regressor.estimators_)
    return pd.DataFrame({'target': [target_gene_name], 'n.estimators': [n_estimators]})


def to_links_df(
    regressor_type, regressor_kwargs, trained_regressor,
    tf_matrix_gene_names, target_gene_name
):
    """
    Returns a pandas dataframe ['tf', 'target', 'importance'] representing inferred 
    regulatory links and their connection strength.
    """

    def pythonic():
        
        feature_importances = to_feature_importances(regressor_type, regressor_kwargs, trained_regressor)
        links_df = pd.DataFrame({'tf': tf_matrix_gene_names, 'importance': feature_importances})
        links_df['target'] = target_gene_name
        clean_links_df = links_df[links_df.importance > 0].sort_values(
            by = 'importance', ascending = False)

        return clean_links_df[['tf', 'target', 'importance']]

    if is_sklearn_regressor(regressor_type):
        return pythonic()
    elif is_xgboost_regressor(regressor_type):
        error('xgb regressor not yet supported')
    else: error('unsupported regressor type: ' + regressor_type)


def clean(tf_matrix, tf_matrix_gene_names, target_gene_name):

    if target_gene_name not in tf_matrix_gene_names:
        clean_tf_matrix = tf_matrix
    else:
        ix = tf_matrix_gene_names.index(target_gene_name)
        if isinstance(tf_matrix, scipy.sparse.spmatrix):
            clean_tf_matrix = scipy.sparse.hstack([tf_matrix[:, :ix], tf_matrix[:, ix+1:]])
        else: clean_tf_matrix = np.delete(tf_matrix, ix, 1)

    clean_tf_names = [tf for tf in tf_matrix_gene_names if tf != target_gene_name]
    assert clean_tf_matrix.shape[1] == len(clean_tf_names)  # sanity check
    return clean_tf_matrix, clean_tf_names


def retry(fn, max_retries = 10, warning_msg = None, fallback_result = None):
    """
    Minimalistic retry strategy to compensate for failures probably caused by a 
    thread-safety bug in scikit-learn:
    * https://github.com/scikit-learn/scikit-learn/issues/2755
    * https://github.com/scikit-learn/scikit-learn/issues/7346
    """

    nr_retries = 0
    result = fallback_result

    for attempt in range(max_retries):
        try: result = fn()
        except Exception as cause:
            nr_retries += 1
            msg_head = '' if warning_msg is None else repr(warning_msg) + ' '
            msg_tail = "retry ({1}/{2}). failure caused by {0}.".format(repr(cause), nr_retries, max_retries)
            warning(msg_head + msg_tail)
        else: break

    return result


def infer_partial_network(
    regressor_type, regressor_kwargs,
    tf_matrix, tf_matrix_gene_names, target_gene_name, target_gene_expression,
    include_meta = False, early_stop_window_length = EARLY_STOP_WINDOW_LENGTH,
    seed = DEMON_SEED
):
    """
    Ties together regressor model training with regulatory links and meta data extraction.

    Parameters
    ----------
    tf_matrix: 
        Numpy matrix. The feature matrix X to use for the regression.

    tf_matrix_gene_names:
        list of transcription factor names corresponding to the columns of the 
        tf_matrix used to train the regression model.

    target_gene_name: 
        The name of the target gene to infer the regulatory links for.

    target_gene_expression:
        The expression profile of the target gene.

    include_meta: 
        Whether to also return the meta information data frame.
    
    early_stop_window_length: 
        Window length of the early stopping monitor.

    seed: 
        (optional) Random seed for the regressors.
    
    Returns
    -------
    if include_meta is True, return links_df, meta_df

    link_df: a pandas dataframe['tf', 'target', 'importance'] containing inferred 
        regulatory links and their connection strength.

    meta_df: a pandas dataframe['target', 'meta', 'value'] containing meta 
        information regarding the trained regression model.
    """

    def fn():

        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(
            tf_matrix, tf_matrix_gene_names, target_gene_name)

        # special case in which only a single tf is passed and the target gene
        # here is the same as the TF (clean_tf_matrix is empty after cleaning):
        if clean_tf_matrix.size == 0:
            warning("cleaned transcriptional factor matrix is empty")
            error("skipping inference of target {}.".format(target_gene_name))

        try:
            trained_regressor = fit_model(
                regressor_type, regressor_kwargs, clean_tf_matrix, target_gene_expression,
                early_stop_window_length, seed
            )
        
        except ValueError as e:
            error("regression for target gene {0} failed. cause {1}.".format(
                target_gene_name, repr(e)))

        links_df = to_links_df(
            regressor_type, regressor_kwargs, trained_regressor, 
            clean_tf_matrix_gene_names, target_gene_name
        )

        if include_meta:
            meta_df = to_meta_df(trained_regressor, target_gene_name)
            return links_df, meta_df
        else: return links_df

    fallback_result = (grn_schema, meta_schema) if include_meta else grn_schema
    return retry(
        fn, fallback_result = fallback_result,
        warning_msg = 'infer_data failed for target {0}'.format(target_gene_name)
    )


def target_gene_indices(gene_names, target_genes):
    """
    Parameters
    ----------
    gene_names
        List of gene names.
    
    target_genes
        Either int (the top n), 'all', or a collection (subset of gene_names).
    
    Returns
    -------
    The (column) indices of the target genes in the expression matrix.
    """

    if isinstance(target_genes, list) and len(target_genes) == 0:
        return []

    if isinstance(target_genes, str) and target_genes.lower() == 'all':
        return list(range(len(gene_names)))

    elif isinstance(target_genes, int):
        top_n = target_genes
        assert top_n > 0
        return list(range(min(top_n, len(gene_names))))

    elif isinstance(target_genes, list):
        if not target_genes:  # target_genes is empty
            return target_genes
        elif all(isinstance(target_gene, str) for target_gene in target_genes):
            return [index for index, gene in enumerate(gene_names) if gene in target_genes]
        elif all(isinstance(target_gene, int) for target_gene in target_genes):
            return target_genes
        else: error("mixed types in target genes.")

    else: error("unable to interpret target_genes.")


class early_stop_monitor:

    def __init__(self, window_length = EARLY_STOP_WINDOW_LENGTH):
        self.window_length = window_length

    def window_boundaries(self, current_round):
        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1
        return lo, hi

    def __call__(self, current_round, regressor, _):
        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else: return False
