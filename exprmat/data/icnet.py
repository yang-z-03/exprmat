
from exprmat import basepath
import pandas as pd
import os


def get_gene_regulatory_network(taxa):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', 'gene-regulatory.feather'))

def get_lr_network(taxa):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', 'ligand-receptors.feather'))

def get_signaling_network(taxa):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', 'signaling.feather'))

def get_gene_regulatory_weights(taxa, version = 'v2'):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', version, 'gene-regulatory.feather'))

def get_lr_weights(taxa, version = 'v2'):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', version, 'lr-significance.feather'))

def get_ligand_target_matrix(taxa, version = 'v2'):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', version, 'ligand-target.feather'))

def get_ligand_tf_matrix(taxa, version = 'v2'):
    return pd.read_feather(os.path.join(basepath, taxa, 'icnet', version, 'ligand-tf.feather'))


def get_weighted_ligand_target_links(ligand, geneset, ligand_target_matrix, n = 250):
    import numpy as np
    ltm_series = ligand_target_matrix[ligand]
    data = np.array(ltm_series.values)
    flat = data.flatten('C')
    indices = np.argsort(flat)[-n:]

    # retrieve indices
    # rows are targets and columns are ligands.
    ligs = []
    targs = []
    weights = []
    for i in indices:
        row = i // len(ligand)
        col = i % len(ligand)
        target = ligand_target_matrix.index[row]
        if target in geneset:
            ligs.append(ligand[col])
            targs.append(ligand_target_matrix.index[row])
            weights.append(flat[i])

    return pd.DataFrame({
        'ligand': ligs,
        'target': targs,
        'weight': weights
    })


def get_weighted_ligand_receptor_links(
    best_upstream_ligands, expressed_receptors, 
    lr_network, weighted_networks_lr_sig
):
    lr_network = lr_network.drop_duplicates(subset = ['from', 'to'])
    weighted_networks_lr = pd.merge(weighted_networks_lr_sig, lr_network, on = ['from', 'to'])

    lr_network_top = lr_network[
        (lr_network['from'].isin(best_upstream_ligands)) & 
        (lr_network['to'].isin(expressed_receptors))
    ].drop_duplicates(subset = ['from','to'])
    best_upstream_receptors = lr_network_top['to'].unique()

    lr_network_top_df_long = weighted_networks_lr[
        (weighted_networks_lr['from'].isin(best_upstream_ligands)) & 
        (weighted_networks_lr['to'].isin(best_upstream_receptors))]
    
    return lr_network_top_df_long
