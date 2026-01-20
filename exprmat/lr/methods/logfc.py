
import numpy as np
from scipy.stats import gmean
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def logfc_score(x):

    mean_logfc = np.mean((x[C.ligand_logfc], x[C.receptor_logfc]), axis = 0)
    return None, mean_logfc


logfc_config = method_config(
    method_name = 'logfc',
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [C.ligand_logfc, C.receptor_logfc],
    fun = logfc_score,
    magnitude = None,
    magnitude_ascending = None,
    specificity = 'lr.logfc',
    specificity_ascending = False,
    permute = False,
    reference = 'Nature Communications, 13(1), pp.1-13'
)

logfc = lr_method(logfc_config)