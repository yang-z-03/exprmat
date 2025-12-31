
from exprmat.data.icnet import (
    get_gene_regulatory_network,
    get_gene_regulatory_weights,
    get_ligand_target_matrix,
    get_ligand_tf_matrix,
    get_lr_network,
    get_lr_weights,
    get_signaling_network,
    get_weighted_ligand_receptor_links,
    get_weighted_ligand_target_links
)

from exprmat.lr.icnet.importance import predict_ligand_activities
from exprmat import info, error
import numpy as np


def nichenet(
    adata,
    taxa,
    receiver, # you must specify the receiver cells
    sender = None,
    identity = 'cell.type',
    foreground = None,
    background = None,
    expression_threshold = 0.001,
    ncpus = 1,
    key_added = 'nichenet'
):
    ltm = get_ligand_target_matrix(taxa)

    if background:
        background = list(set(background))
        info(f'using user-specified background gene set of size {len(background)}')
    
    else: background = adata.var['gene'].tolist()
    background = list(set(background).intersection(set(ltm.index.tolist())))
    info(f'background gene set constructed with size {len(background)}')
    
    if foreground:
        foreground = list(set(foreground))
        info(f'using user-specified foreground gene set of size {len(foreground)}')
    else: error('foreground gene set must be specified')
    
    expressed_genes = adata.var['gene'].loc[
        np.array((adata.X.sum(0) / adata.X.shape[0]) > expression_threshold)[0]].tolist()
    rec = adata[[x in receiver for x in adata.obs[identity]], :]
    target_expressed_genes = rec.var['gene'].loc[np.array(
        (rec.X.sum(0) / rec.X.shape[0]) > expression_threshold)[0]].tolist()
    
    lr_network = get_lr_network(taxa)
    all_receptors = lr_network['to'].unique().tolist()
    expressed_receptors = list(set(all_receptors).intersection(set(target_expressed_genes)))
    potential_ligands = [x for x, y in zip(lr_network['from'], lr_network['to']) if y in expressed_genes]

    if sender:
        send = adata[[x in sender for x in adata.obs[identity]], :]
        sender_expressed_genes = send.var['gene'].loc[np.array(
            (send.X.sum(0) / send.X.shape[0]) > expression_threshold)[0]].tolist()
        potential_ligands = list(set(potential_ligands).intersection(set(sender_expressed_genes)))
        info(f'{len(potential_ligands)} potential ligands filtered by sender cells expression')
    else: 
        potential_ligands = list(set(potential_ligands))
        info(f'using {len(potential_ligands)}(all) potential ligands')

    info(f'using {len(expressed_receptors)} expressed receptors in receivers')
    
    ligact = predict_ligand_activities(
        geneset = foreground,
        background_expressed_genes = background,
        ligand_target_matrix = ltm.values,
        ligand_names = ltm.columns.tolist(),
        target_names = ltm.index.tolist(),
        potential_ligands = potential_ligands,
        ncpus = ncpus
    )

    adata.uns[key_added] = {
        'foreground': foreground,
        'background': background,
        'taxa': taxa,
        'activities': ligact.sort_values(['aupr.adj'], ascending = False)
    }

    pass


def nichenet_infer_targets(adata, key_nichenet, ligands = 10, n_targets = 250):

    nnet = adata.uns[key_nichenet]
    ltm = get_ligand_target_matrix(nnet['taxa'])
    ltl = get_weighted_ligand_target_links(
        ligand = ligands if isinstance(ligands, list) else nnet['activities'].head(ligands)['ligand'].tolist(),
        geneset = nnet['foreground'],
        ligand_target_matrix = ltm,
        n = n_targets
    )

    adata.uns[key_nichenet]['ligand_targets'] = ltl