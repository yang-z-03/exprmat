
import numpy as np
from exprmat.ansi import error, warning


# expansion index for cluster.
# maximal value if the cluster is `dominant` for a single clonotype e.g. mait cells.

def clone_cluster_matrix(adata, clonotype = 'clone.id', cluster = 'leiden'):

    clono_ids = adata.obs[clonotype].value_counts().index.tolist()
    clust_ids = adata.obs[cluster].value_counts().index.tolist()
    clono_col = adata.obs[clonotype]
    clust_col = adata.obs[cluster]
    mat = np.ndarray(shape = (len(clono_ids), len(clust_ids)), dtype = np.float32)
    
    for j, cs in enumerate(clust_ids):
        specific_cluster = clono_col[clust_col == cs].copy()
        clono_stat = specific_cluster.value_counts()

        for i, cl in enumerate(clono_ids):
            if cl in clono_stat.index: mat[i, j] = clono_stat[cl]
            else: mat[i, j] = 0

    return mat, clono_ids, clust_ids


def expansion(adata, clonotype = 'clone.id', cluster = 'leiden', key = 'tcr.cluster.expansion'):
    
    mat, clono_ids, clust_ids = clone_cluster_matrix(adata, clonotype = clonotype, cluster = cluster)
    adata.obs[key] = 0

    for j, cs in enumerate(clust_ids):
        props = mat[:, j] / np.sum(mat[:, j])
        props = props[np.nonzero(mat[:, j])]
        
        if np.sum(mat[:, j]) < 5: 
            adata.obs.loc[adata.obs[cluster] == cs, key] = 0
        else:
            shannon_e = - np.sum(props * np.log2(props))
            adata.obs.loc[adata.obs[cluster] == cs, key] = 1 - (shannon_e / np.log2(len(props)))


def plasticity(adata, clonotype = 'clone.id', cluster = 'leiden', key = 'tcr.cluster.plasticity'):

    mat, clono_ids, clust_ids = clone_cluster_matrix(adata, clonotype = clonotype, cluster = cluster)
    adata.obs[key] = 0

    for j, cs in enumerate(clust_ids):
        # select all clones that present in this cluster
        submat = mat[mat[:, j] > 0, :]
        submat = (submat.T / submat.sum(axis = 1)).T # row normalize
        overall = submat.sum(axis = 0)
        overall = overall / overall.sum()
        overall = overall[np.nonzero(overall)]
        # completely deterministic, 0. completely random, 1.

        if np.sum(mat[:, j]) < 5: 
            adata.obs.loc[adata.obs[cluster] == cs, key] = 0
        else:
            shannon_e = - np.sum(overall * np.log2(overall))
            adata.obs.loc[adata.obs[cluster] == cs, key] = shannon_e


def pairwise_transition(
    adata, base,  clonotype = 'clone.id', cluster = 'leiden', 
    key = 'tcr.cluster.ptrans'
):

    mat, clono_ids, clust_ids = clone_cluster_matrix(adata, clonotype = clonotype, cluster = cluster)
    adata.obs[key] = 0
    baseid = clust_ids.index(base)

    for j, cs in enumerate(clust_ids):
        if j == baseid: continue

        # select all clones that present in this cluster
        submat = mat[mat[:, j] + mat[:, baseid] > 0, :]
        submat = submat[:, [baseid, j]]
        either_zero = (submat[:, 0] == 0) | (submat[:, 1] == 0)
        weights = submat[:, 0] + submat[:, 1]
        weights = weights / weights.sum()
        submat = (submat.T / submat.sum(axis = 1)).T # row normalize
        
        shannon_e = -(submat[:, 0] * np.log2(submat[:, 0]) + submat[:, 1] * np.log2(submat[:, 1]))
        shannon_e[either_zero] = 0
        adata.obs.loc[adata.obs[cluster] == cs, key] = np.sum(shannon_e * weights)


# per clone statistics

def transition(adata, clonotype = 'clone.id', cluster = 'leiden', key = 'tcr.clone.trans'):

    mat, clono_ids, clust_ids = clone_cluster_matrix(adata, clonotype = clonotype, cluster = cluster)
    adata.obs[key] = 0

    for i, cl in enumerate(clono_ids):
        props = mat[i, :] / np.sum(mat[i, :])
        props = props[np.nonzero(mat[i, :])]
        shannon_e = - np.sum(props * np.log2(props))
        adata.obs.loc[adata.obs[clonotype] == cl, key] = shannon_e




def migration(adata, clonotype = 'clone.id', cluster = 'tissue', key = 'tcr.clone.migr'):
    # completely the same.
    transition(adata, clonotype = clonotype, cluster = cluster, key = key)