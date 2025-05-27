
import numpy as np
from scipy.stats import gmean
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def spec_weight(ligand_means, ligand_means_sums, receptor_means, receptor_means_sums):
    s_lig = (ligand_means / ligand_means_sums)
    s_rec = (receptor_means / receptor_means_sums)
    return s_lig * s_rec


def natmi_score(x) -> tuple:

    # magnitude
    expr_prod = x[D.ligand_means] * x[D.receptor_means]

    # specificity
    spec = spec_weight(
        x[D.ligand_means], x[C.ligand_means_sums],
        x[D.receptor_means], x[C.receptor_means_sums])

    return expr_prod, spec


natmi_config = method_config(
    method_name = "natmi",
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [C.ligand_means_sums, C.receptor_means_sums],
    fun = natmi_score,
    magnitude = 'expr.prod',
    magnitude_ascending = False,
    specificity = 'spec.weight',
    specificity_ascending = False,
    permute = False,
    reference = 'Nature communications, 11(1), pp.1-11'
)

natmi = lr_method(natmi_config)