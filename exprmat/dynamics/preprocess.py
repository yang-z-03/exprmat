
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

from exprmat.preprocessing import normalize, log1p
from exprmat.dynamics.generank import get_mean_var
from exprmat.dynamics.linreg import sum, multiply
from exprmat.dynamics.moments import not_yet_normalized
from exprmat.ansi import error, warning, info
from exprmat.utils import choose_layer


def get_size(adata, modality = None):

    X = choose_layer(adata, layer = modality)
    return sum(X, axis = 1)


def set_initial_size(adata, layers = None) -> None:
    
    if layers is None: layers = ["unspliced", "spliced"]

    layers = [
        layer for layer in layers if layer in adata.layers.keys()
        and f"n.umi.{layer}" not in adata.obs.keys()
    ]
    for layer in layers: adata.obs[f"n.umi.{layer}"] = get_size(adata, layer)
    if "n.umi" not in adata.obs.keys(): adata.obs["n.umi"] = get_size(adata)


def get_initial_size(adata, layer = None, by_total_size = False):
    
    if by_total_size:
        sizes = [
            adata.obs[f"n.umi.{layer}"]
            for layer in {"spliced", "unspliced"}
            if f"n.umi.{layer}" in adata.obs.keys()
        ]
        return np.sum(sizes, axis = 0)
    
    elif layer in adata.layers.keys():
        return (
            np.array(adata.obs[f"n.umi.{layer}"])
            if f"n.umi.{layer}" in adata.obs.keys()
            else get_size(adata, layer)
        )
    
    elif layer is None or layer == "X":
        return (
            np.array(adata.obs["n.umi"])
            if "n.umi" in adata.obs.keys()
            else get_size(adata)
        )
    
    else: return None


def filter_matrix(X, min_counts = None, min_cells = None, max_counts = None, max_cells = None):
    
    counts = (
        sum(X, axis = 0)
        if (min_counts is not None or max_counts is not None)
        else sum(X > 0, axis = 0)
    )

    lb = (
        min_counts
        if min_counts is not None
        else min_cells
        if min_cells is not None
        else -np.inf
    )

    ub = (
        max_counts
        if max_counts is not None
        else max_cells
        if max_cells is not None
        else np.inf
    )

    return (lb <= counts) & (counts <= ub), counts


def filter_genes(
    adata,
    min_counts = None,
    min_cells = None,
    max_counts = None,
    max_cells = None,
    min_counts_u = None,
    min_cells_u = None,
    max_counts_u = None,
    max_cells_u = None,
    min_shared_counts = None,
    min_shared_cells = None,
    retain_genes = None
):
    """
    Filter genes based on number of cells or counts.

    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells. Only provide one of the optional parameters 
    `min_counts`, `min_cells`, `max_counts`, `max_cells` per call.
    """

    # set initial cell sizes before filtering
    set_initial_size(adata)

    layers = [layer for layer in ["spliced", "unspliced"] if layer in adata.layers.keys()]
    if min_shared_counts is not None or min_shared_cells is not None:
        layers.extend(["shared"])

    for layer in layers:

        if layer == "spliced":
            _min_counts, _min_cells, _max_counts, _max_cells = (
                min_counts,
                min_cells,
                max_counts,
                max_cells,
            )

        elif layer == "unspliced":
            _min_counts, _min_cells, _max_counts, _max_cells = (
                min_counts_u,
                min_cells_u,
                max_counts_u,
                max_cells_u,
            )

        else:  # shared counts/cells
            _min_counts, _min_cells, _max_counts, _max_cells = (
                min_shared_counts,
                min_shared_cells,
                None,
                None,
            )

        if layer in adata.layers.keys():
            X = adata.layers[layer]
        else:  # shared counts/cells
            Xs, Xu = adata.layers["spliced"], adata.layers["unspliced"]
            nonzeros = multiply(Xs > 0, Xu > 0)
            X = multiply(nonzeros, Xs) + multiply(nonzeros, Xu)

        gene_subset = np.ones(adata.n_vars, dtype = bool)

        if _min_counts is not None or _max_counts is not None:
            gene_subset &= filter_matrix(X, min_counts=_min_counts, max_counts=_max_counts)[0]

        if _min_cells is not None or _max_cells is not None:
            gene_subset &= filter_matrix(X, min_cells=_min_cells, max_cells=_max_cells)[0]

        if retain_genes is not None:
            if isinstance(retain_genes, str): retain_genes = [retain_genes]
            gene_subset |= adata.var_names.isin(retain_genes)

        adata._inplace_subset_var(gene_subset)
        s = np.sum(~ gene_subset)
        if s > 0: info(f"Filtered out {s} genes that are detected")

    return



def csr_vcorrcoef(X, y):

    mu_x = np.ravel(np.mean(X, axis=-1))
    mu_y = np.ravel(np.mean(y, axis=-1))
    nom = X.dot(y) - X.dot(np.repeat(mu_y, len(y))) - mu_x * np.sum(y - mu_y)

    if X.ndim == 1:
        n_features = len(X)
    else:
        n_features = X.shape[1]

    denom_x = (
        np.ravel(np.sum(X.multiply(X), axis=-1))
        if issparse(X)
        else np.sum(X * X, axis=-1)
    )
    denom_x = denom_x - 2 * np.ravel(np.sum(X, axis=-1)) * mu_x + n_features * mu_x**2
    denom_y = (
        np.ravel(np.sum(y * y, axis=-1))
        - 2 * (np.ravel(np.sum(y, axis=-1)) * mu_y)
        + n_features * mu_y**2
    )

    return nom / np.sqrt(denom_x * denom_y)


def counts_per_cell_quantile(X, max_proportion_per_cell = 0.05, counts_per_cell = None):

    if counts_per_cell is None:
        counts_per_cell = sum(X, axis=1)
    gene_subset = np.all(
        X <= counts_per_cell[:, None] * max_proportion_per_cell, axis=0
    )
    if issparse(X):
        gene_subset = gene_subset.A1
    return sum(X[:, gene_subset], axis=1)


def check_if_valid_dtype(adata, layer="X"):

    X = adata.X if layer == "X" else adata.layers[layer]
    if "int" in X.dtype.name:
        if layer == "X":
            adata.X = adata.X.astype(np.float32)
        elif layer in adata.layers.keys():
            adata.layers[layer] = adata.layers[layer].astype(np.float32)


# normalize to the median of cell n_obs.
def normalize_per_cell(
    data,
    counts_per_cell_after = None,
    counts_per_cell = None,
    key_n_counts = None,
    max_proportion_per_cell = None,
    use_initial_size = True,
    layers = None,
    enforce = None,
    copy = False,
):
    adata = data.copy() if copy else data
    if layers is None: layers = ["spliced", "unspliced"]
    elif layers == "all": layers = adata.layers.keys()
    elif isinstance(layers, str): layers = [layers]
    layers = [layer for layer in layers if layer in adata.layers.keys()]
    modified_layers = []

    if isinstance(counts_per_cell, str):
        if counts_per_cell not in adata.obs.keys(): set_initial_size(adata, layers)
        counts_per_cell = (
            adata.obs[counts_per_cell].values
            if counts_per_cell in adata.obs.keys()
            else None
        )

    for layer in layers:
        
        check_if_valid_dtype(adata, layer)
        X = adata.X if layer == "X" else adata.layers[layer]

        if not_yet_normalized(X) or enforce:
            counts = (
                counts_per_cell
                if counts_per_cell is not None
                else get_initial_size(adata, layer)
                if use_initial_size
                else get_size(adata, layer)
            )
            if max_proportion_per_cell is not None and \
                (0 < max_proportion_per_cell < 1):
                counts = counts_per_cell_quantile(X, max_proportion_per_cell, counts)
            
            # equivalent to normalize_per_cell(X, counts_per_cell_after, counts)
            counts_after = (
                np.median(counts)
                if counts_per_cell_after is None
                else counts_per_cell_after
            )

            counts_after += counts_after == 0
            counts = counts / counts_after
            counts += counts == 0  # to avoid division by zero

            if issparse(X): sparsefuncs.inplace_row_scale(X, 1 / counts)
            else: X /= np.array(counts[:, None])
            modified_layers.append(layer)

            if (
                layer == "X"
                and "gene.count.corr" not in adata.var.keys()
                and X.shape[-1] > 3e3
            ):
                try:
                    adata.var["gene.count.corr"] = np.round(
                        csr_vcorrcoef(X.T, np.ravel((X > 0).sum(1))), 4)
                except ValueError: pass

        else:
            warning(f"did not normalize {layer} as it looks processed already.") 
            warning(f"to enforce normalization, set `enforce = True`.")

    adata.obs["n.umi" if key_n_counts is None else key_n_counts] = get_size(adata)
    if len(modified_layers) > 0:
        info("normalized count data: " + f"{', '.join(modified_layers)}.")

    return adata if copy else None


def filter_and_normalize(
    adata,
    min_counts = None,
    min_counts_u = None,
    min_cells = None,
    min_cells_u = None,
    min_shared_counts = None,
    min_shared_cells = None,
    retain_genes = None,
    # do not change X by default.
    layers_normalize = ['spliced', 'unspliced'],
    **kwargs,
):
    if "spliced" not in adata.layers.keys() or "unspliced" not in adata.layers.keys():
        error("could not find spliced / unspliced counts.")

    # since the normalization operation will take inplace.
    if 'spliced.counts' not in adata.layers.keys():
        adata.layers['spliced.counts'] = adata.layers['spliced'].copy()

    if 'unspliced.counts' not in adata.layers.keys():
        adata.layers['unspliced.counts'] = adata.layers['unspliced'].copy()

    filter_genes(
        adata,
        min_counts = min_counts,
        min_counts_u = min_counts_u,
        min_cells = min_cells,
        min_cells_u = min_cells_u,
        min_shared_counts = min_shared_counts,
        min_shared_cells = min_shared_cells,
        retain_genes = retain_genes,
    )

    if layers_normalize is not None and "enforce" not in kwargs:
        kwargs["enforce"] = True
    normalize_per_cell(adata, layers = layers_normalize, **kwargs)
    return adata