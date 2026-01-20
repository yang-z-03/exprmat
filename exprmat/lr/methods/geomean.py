
import numpy as np
from scipy.stats import gmean
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def geomean_score(x, perm_stats) -> tuple:

    lr_gmeans = gmean((x[D.ligand_means].values, x[D.receptor_means].values), axis = 0)
    gmean_pvals = calculate_pvals(lr_gmeans, perm_stats, gmean)
    return lr_gmeans, gmean_pvals


geomean_config = method_config(
    method_name = 'geomean',
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [],
    fun = geomean_score,
    magnitude = "lr.geomean",
    magnitude_ascending = False,
    specificity = "geomean.p",
    specificity_ascending = True,
    permute = True,
    reference = "Geometric mean"
)

geomean = lr_method(_method = geomean_config)