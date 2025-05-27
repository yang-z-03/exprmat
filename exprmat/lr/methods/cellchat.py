
import numpy as np
from exprmat.lr.method import lr_method, method_config
from exprmat.lr.utils import calculate_pvals
from exprmat.lr.utils import common_method_columns as C


def lr_probability(perm_stats, axis=0):
    lr_prob = np.prod(perm_stats, axis = axis)
    return lr_prob / (cellchat._kh + lr_prob)


def cellchat_score(x, perm_stats) -> tuple:
    lr_prob = lr_probability((x[C.ligand_trimean].values, x[C.receptor_trimean].values))
    cellchat_pvals = calculate_pvals(lr_prob, perm_stats, lr_probability)
    return lr_prob, cellchat_pvals


cellchat_config = method_config(
    method_name = "cellchat",
    complex_cols = [C.ligand_trimean, C.receptor_trimean],
    add_cols = [C.mat_max],
    fun = cellchat_score,
    magnitude = "lr.probs",
    magnitude_ascending = False,
    specificity = "cellchat.p",
    specificity_ascending = True,
    permute = True,
    reference = "Nature communications, 12(1), pp.1-20"
)

cellchat = lr_method(cellchat_config)
cellchat._kh = 0.5
