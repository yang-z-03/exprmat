
import argparse
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import PurePath
import pandas as pd
from rich.progress import track

from exprmat.utils import choose_layer
from exprmat.data.tf import get_tfs
from exprmat.ansi import info, warning, error
from exprmat.grn.regressors import (
    EARLY_STOP_WINDOW_LENGTH,
    kwargs_random_forest,
    kwargs_stochastic_gbm,
    infer_partial_network,
    target_gene_indices,
    to_tf_matrix,
)


def run_infer_partial_network(
    target_gene_index,
    gene_names,
    ex_matrix,
    tf_matrix,
    tf_matrix_gene_names,
    method_params,
    seed,
):
    target_gene_name = gene_names[target_gene_index]
    target_gene_expression = ex_matrix[:, target_gene_index]

    n = infer_partial_network(
        regressor_type = method_params[0],
        regressor_kwargs = method_params[1],
        tf_matrix = tf_matrix,
        tf_matrix_gene_names = tf_matrix_gene_names,
        target_gene_name = target_gene_name,
        target_gene_expression = target_gene_expression,
        include_meta = False,
        early_stop_window_length = EARLY_STOP_WINDOW_LENGTH,
        seed = seed,
    )
    return n


def adjacencies(
    adata, layer = 'counts', gene = 'gene', 
    method = 'grnboost2', taxa = 'hsa', ncpus = 1, seed = 42
):
    
    if method == "grnboost2": method_params = ["GBM", kwargs_stochastic_gbm]
    elif method == "genie3": method_params = ["RF", kwargs_random_forest]

    # construct a cell by gene matrix in pandas dataframe
    ex_matrix = choose_layer(adata, use_raw = False, layer = layer)
    gene_names = adata.var[gene].tolist() if gene is not None else adata.var_names.tolist()

    tf_names = get_tfs(taxa)
    info(f"loaded {len(tf_names)} transcriptional factors.")
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(ex_matrix, gene_names, tf_names)
    info(f"starting {method} using {ncpus} processes")
    start_time = time.time()

    with Pool(ncpus) as p:
        adjs = list(
            track(
                p.imap(
                    partial(
                        run_infer_partial_network,
                        gene_names = gene_names,
                        ex_matrix = ex_matrix,
                        tf_matrix = tf_matrix,
                        tf_matrix_gene_names = tf_matrix_gene_names,
                        method_params = method_params,
                        seed = seed,
                    ),
                    target_gene_indices(gene_names, target_genes = "all"),
                    chunksize = 1,
                ),
                total = len(gene_names),
                description = 'inferring partial network '
            )
        )

    adj = pd.concat(adjs).sort_values(by = "importance", ascending = False)
    end_time = time.time()
    info(f"finished in {end_time - start_time} seconds.")
    return adj
