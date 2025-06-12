
import os
import numpy as np
import torch

from exprmat.ansi import error, warning, info
from exprmat.reader.experiment import experiment


def construct_atlas(
    expm: experiment, 
    key_counts = 'counts',
    key_batch = 'batch',
    batch_cell_filter = 50,

    # scvi model settings
    scvi_n_epoch = None,
    scvi_n_latent = 10,
    scvi_n_hidden = 128,
    scvi_n_layers = 1,
    scvi_dropout_rate = 0.1,
    scvi_dispersion = 'gene',
    scvi_gene_likelihood = 'zinb',
    scvi_latent_distrib = 'normal',
    scvi_key = 'scvi',

    annotation = 'cell.type',
    scanvi_key = 'scanvi',
    scanvi_unlabel = 'unknown',
    scanvi_max_epochs = 30,
    scanvi_samples_per_label = 200,
):
    
    import warnings
    warnings.filterwarnings('ignore')
    torch.set_float32_matmul_precision('high')

    atlas = expm.mudata['rna']
    savename = expm.subset if expm.subset is not None else 'integrated'
    path = os.path.join(expm.directory, 'scvi', savename)
    
    if os.path.exists(path):
        error('the destination path `scvi/{savename}` is already occupied.')
    else: os.makedirs(os.path.join(expm.directory, 'scvi'), exist_ok = True)


    # extract count matrix.
    import anndata
    counts = anndata.AnnData(
        X = atlas.layers[key_counts] \
            if key_counts is not None else atlas.X
    )

    counts.obs['batch'] = atlas.obs[key_batch].tolist() \
        if key_batch in atlas.obs.keys() else 'whole'
    counts.obs['annotation'] = atlas.obs[annotation].tolist()
    counts.var_names = atlas.var_names.tolist()


    # build scvi model
    import scvi
    
    # we will remove all data with < batch_cell_filter cell detection.
    mapping = {}
    names = counts.obs['batch'].value_counts().index.tolist()
    values = counts.obs['batch'].value_counts().tolist()
    n_outlier_sample = 0
    for n, v in zip(names, values):
        if v > batch_cell_filter: mapping[n] = n
        else: 
            mapping[n] = 'outliers'
            n_outlier_sample += 1
    
    warning(f'{n_outlier_sample} samples is removed due to small sample size.')
    batch = counts.obs['batch'].tolist()
    for i in range(len(batch)):
        batch[i] = mapping[batch[i]]
    counts.obs['batch'] = batch

    # extract metadata file.
    import scipy.sparse as sparse
    n_cells, n_genes = counts.X.shape
    metadata = anndata.AnnData(
        X = sparse.csr_matrix((n_cells, n_genes), dtype = np.float32),
        obs = counts.obs, var = counts.var
    )

    scvi.model.SCVI.setup_anndata(counts, batch_key = 'batch')
    model = scvi.model.SCVI(
        counts, 
        n_hidden = scvi_n_hidden, 
        n_latent = scvi_n_latent, 
        n_layers = scvi_n_layers,
        dropout_rate = scvi_dropout_rate,
        dispersion = scvi_dispersion,
        gene_likelihood = scvi_gene_likelihood,
        latent_distribution = scvi_latent_distrib
    )

    max_epochs_scvi = np.min([round((20000 / counts.n_obs) * 400), 400]) \
        if scvi_n_epoch is None else scvi_n_epoch
    info(f'will train {max_epochs_scvi} epochs.')
    model.train(max_epochs = int(max_epochs_scvi), early_stopping = True)
    scvi_pc = model.get_latent_representation()
    info(f'trained scvi model saved.')
    model.save(path)

    metadata.obsm[scvi_key] = scvi_pc
    metadata.uns['atlas'] = {
        'batch': 'batch',
        'annotation': 'annotation',
        'latent': scvi_n_latent,
        'scvi': scvi_key,
        'scanvi': scanvi_key,
    }

    atlas.obsm[scvi_key] = scvi_pc
    atlas.obs['atlas.batch'] = metadata.obs['batch'].tolist()
    atlas.uns['atlas'] = {
        'batch': 'atlas.batch',
        'annotation': annotation,
        'latent': scvi_n_latent,
        'scvi': scvi_key,
        'scanvi': scanvi_key,
    }

    
    # build scanvi model
    info(f'training scanvi model on label `{annotation}`')
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        unlabeled_category = scanvi_unlabel,
        labels_key = 'annotation'
    )

    scanvi_model.train(
        max_epochs = scanvi_max_epochs, 
        n_samples_per_label = scanvi_samples_per_label
    )

    atlas.obsm[scanvi_key] = scanvi_model.get_latent_representation()
    metadata.obsm[scanvi_key] = atlas.obsm[scanvi_key]

    info(f'trained scanvi model saved.')
    scanvi_model.save(path, prefix = 'scanvi.')
    info(f'saving metadata of atlas as a light version.')
    metadata.write_h5ad(os.path.join(expm.directory, 'scvi', savename, 'metadata.h5ad'))
    warnings.filterwarnings('default')


def is_atlas(expm: experiment):

    expm.check_merged('rna')

    if 'atlas' not in expm.mudata['rna'].uns.keys():
        warning(f'do not contain an `atlas` key in uns slot. build atlas first.')
        return False
    
    atlas = expm.mudata['rna'].uns['atlas']
    if atlas['batch'] not in expm.mudata['rna'].obs.keys():
        warning(f'`{atlas["batch"]}` do not exist.')
        return False
    
    if atlas['annotation'] not in expm.mudata['rna'].obs.keys():
        warning(f'`{atlas["annotation"]}` do not exist.')
        return False

    if atlas['scvi'] not in expm.mudata['rna'].obsm.keys():
        warning(f'`{atlas["scvi"]}` do not exist.')
        return False

    if atlas['scanvi'] not in expm.mudata['rna'].obsm.keys():
        warning(f'`{atlas["scanvi"]}` do not exist.')
        return False

    return True


def embed(
    atlas: experiment, 
    expm: experiment,
    
    counts_key = 'counts',
    batch_key = 'batch',
    key_query_latent = 'scvi',
    scvi_epoch_reduction = 3,
):

    import scvi
    import anndata

    input = expm.mudata['rna']
    query = anndata.AnnData(
        X = input.X.copy() if counts_key is None 
        else input.layers[counts_key].copy()
    )

    query.obs_names = input.obs_names
    query.var_names = input.var_names
    query.obs['annotation'] = 'unknown'
    assert query.var.index.is_unique
    assert query.obs.index.is_unique

    # the scvi requires obs[batch] and var names to be .ugene.
    # so we transform to adapt to it.
    
    if batch_key is not None:
        assert batch_key in input.obs.keys()
        query.obs['batch'] = input.obs[batch_key].tolist()
    else: 
        warning(f'do not supply the batch key. assume they all came from the same batch.')
        query.obs['batch'] = 'whole'
        

    info(f'preparing query data ...')
    scvi_path = os.path.join(
        atlas.directory, 'scvi', 
        atlas.subset if atlas.subset is not None else 'integrated'
    )

    scvi.model.SCVI.prepare_query_anndata(query, scvi_path)
    info(f'constructing query model for batch correction ...')
    query_model = scvi.model.SCVI.load_query_data(query, scvi_path)

    max_epochs_scvi = np.min([round((20000 / query.n_obs) * 400), 400]) // scvi_epoch_reduction
    info(f'will automatically train {max_epochs_scvi} epochs ...')
    query_model.train(max_epochs = int(max_epochs_scvi), plan_kwargs = { 'weight_decay': 0.0 })

    info(f'getting latent representations ...')
    query.obsm[key_query_latent] = query_model.get_latent_representation()
    input.obsm[key_query_latent] = query.obsm[key_query_latent]
    
    return query