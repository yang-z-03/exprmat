
import numpy as np
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def connectome_score(x) -> tuple:

    # magnitude
    expr_prod = x[D.ligand_means].values * x[D.receptor_means].values

    # specificity
    scaled_weight = np.mean(
        (x[C.ligand_zscores].values, x[C.receptor_zscores].values), 
        axis = 0
    )

    return expr_prod, scaled_weight


connectome_config = method_config(
    method_name = "connectome",
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [C.ligand_zscores, C.receptor_zscores],
    fun = connectome_score,
    magnitude = 'expr.prod',
    magnitude_ascending = False,
    specificity = 'scaled.weight',
    specificity_ascending = False,
    permute = False,
    reference = 'Scientific reports, 12(1), pp.1-12'
)

connectome = lr_method(_method = connectome_config)
