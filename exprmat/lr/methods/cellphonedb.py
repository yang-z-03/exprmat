
import numpy as np
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def cpdb_mean(a, axis = 0):
    return np.mean(a, axis = axis)


def cpdb_score(x, perm_stats) -> tuple:
    
    zero_msk = ((x[D.ligand_means] == 0) | (x[D.receptor_means] == 0))
    lr_means = cpdb_mean((x[D.ligand_means].values, x[D.receptor_means].values))
    lr_means[zero_msk] = 0
    cpdb_pvals = calculate_pvals(lr_means, perm_stats, cpdb_mean)
    return lr_means, cpdb_pvals


cellphonedb_config = method_config(
    method_name = "cellphonedb",
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [],
    fun = cpdb_score,
    magnitude = "lr.means",
    magnitude_ascending = False,
    specificity = "cellphonedb.p",
    specificity_ascending = True,
    permute = True,
    reference = "Nature protocols, 15(4), pp.1484-1506"
)

cellphonedb = lr_method(
    _method = cellphonedb_config
)