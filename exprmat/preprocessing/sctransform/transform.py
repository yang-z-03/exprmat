
import numpy as np
import pandas as pd
import patsy
from exprmat.ansi import error, warning
from exprmat.preprocessing.sctransform import estimate
from exprmat.preprocessing.sctransform import ksmooth
from exprmat.preprocessing.sctransform import residual


def vst(
    umi,
    cell_attr = None,
    latent_var = ["log.umi"],
    method = "poisson",
    res_clip_range = None,
    min_cells = 5,
    residual_type = "pearson",
    min_variance = float("-inf"),
    bw_adjust = 3,
    gmean_eps = 1
):
    """
    Regress the latent variables and transform the raw UMI counts into variance stablized
    scaled intensities. This is implemented in Seurat's sctransform routine. However,
    handling batch variables are not implemented yet. It is true that even if the 
    batch variable correction was added in Seurat, incorporating such features here merely
    consider the generalized linear transformation and is not necessary in some cases for
    non-linear systematic biases.

    We recommended to use the reversely calculated 'corrected' counts (a normalized linear)
    representation to run non-linear integration after it.

    Steps of the algorithm:

    1. Subset to 2k genes using density-based weightings
    2. For each gene, fit a Poisson model to estimate B coefficients
       - Then estimate the theta parameter via machine learning regression
    3. Regularize parameters (offset and slope) via kernal function
       - Find gene means (geometric mean)
       - Choose kernel bandwidth
    4. Transform UMI counts into Pearson residuals
       - Clip residuals to maximum value of sqrt N_Cells
    """


    if method != "poisson":
        error("only method = 'poisson' is supported")

    if residual_type != "pearson":
        error("only residual_type = 'pearson' supported")

    # Populate the cell attr with known attributes

    if cell_attr is None:
        cell_attr = pd.DataFrame(index = umi.columns)
    else:
        assert (cell_attr.index == umi.columns).all(), "cell_attr rows must match umi columns"
        cell_attr = cell_attr.copy()

    if "n.umi" in latent_var and "n.umi" not in cell_attr.columns:
        cell_attr["n.umi"] = umi.sum(axis = 0)

    if "n.gene" in latent_var and "n.gene" not in cell_attr.columns:
        cell_attr["n.gene"] = (umi > 0).sum(axis = 0)

    if "log.umi" in latent_var and "log.umi" not in cell_attr.columns:
        cell_attr["log.umi"] = np.log10(umi.sum(axis = 0))

    if "log.gene" in latent_var and "log.gene" not in cell_attr.columns:
        cell_attr["log.gene"] = np.log10((umi > 0).sum(axis = 0))

    if "umi.per.gene" in latent_var and "umi.per.gene" not in cell_attr.columns:
        cell_attr["umi.per.gene"] = umi.sum(axis = 0) / (umi > 0).sum(axis = 0)

    if "log.umi.per.gene" in latent_var and "log.umi.per.gene" not in cell_attr.columns:
        cell_attr["log.umi.per.gene"] = np.log10(umi.sum(axis = 0) / (umi > 0).sum(axis = 0))

    assert all([x in cell_attr.columns for x in latent_var]), "not all latent_var in cell_attr"

    # Subset genes using min_cells
    genes_cell_count = (umi > 0).sum(axis = 1)
    umi = umi.loc[genes_cell_count >= min_cells, ]

    genes_log_gmean = np.log10(
        np.exp(np.log(umi + gmean_eps).mean(axis = 1)) - gmean_eps
    )

    # Create model matrix
    model_matrix = patsy.dmatrix(" + ".join(latent_var), cell_attr)

    # Substitute cells for step1
    genes_log_gmean_step1 = genes_log_gmean

    model_pars = estimate.estimate_parameters_all_genes(umi, model_matrix)

    model_pars_fit, outliers = ksmooth.reg_model_pars(
        model_pars, genes_log_gmean_step1, genes_log_gmean, cell_attr,
        None, umi, bw_adjust, gmean_eps
    )

    residuals = residual.compute_residuals(
        umi, model_matrix, model_pars_fit, res_clip_range, min_variance
    )

    return residuals, model_pars_fit, res_clip_range
