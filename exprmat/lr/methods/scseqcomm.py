
import numpy as np
from scipy.stats import gmean
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C
from exprmat.lr.utils import default_common_columns as D


def interac_score(x):
    inter_score = np.minimum(x[C.ligand_cdf], x[C.receptor_cdf])
    return inter_score, None


scseqcomm_config = method_config(
    method_name = "scseqcomm",
    complex_cols = [D.ligand_means, D.receptor_means],
    add_cols = [C.ligand_cdf, C.receptor_cdf],
    fun = interac_score,
    magnitude = "inter.score",
    magnitude_ascending = False,
    specificity = None,
    specificity_ascending = None,
    permute = False,
    reference = "Bioinformatics, 38(7), pp.1920-1929"
)

scseqcomm = lr_method(scseqcomm_config)
