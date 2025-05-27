
import numpy as np
from scipy.stats import gmean
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def sca_score(x):
    lr_sqrt = np.sqrt(x[D.ligand_means]) * np.sqrt(x[D.receptor_means])
    denominator = (lr_sqrt + x[C.mat_mean])
    return lr_sqrt / denominator, None


scsr_config = method_config(
    method_name = "singlecellsignalr",
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [C.mat_mean],
    fun = sca_score,
    magnitude = 'sca.score',
    magnitude_ascending = False,
    specificity = None,
    specificity_ascending = None,
    permute = False,
    reference = 'Nucleic Acids Research, 48(10), pp.e55-e55'
)

singlecellsignalr = lr_method(scsr_config)