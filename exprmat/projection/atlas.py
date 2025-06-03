
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

    mde_key = 'mde'
):

    atlas = expm.mudata['rna']
    savename = expm.subset if expm.subset is not None else 'integrated'
    path = os.path.join(expm.directory, 'scvi', savename)
    
    if os.path.exists(path):
        error('the destination path `scvi/{savename}` is already occupied.')
    else: os.makedirs(os.path.join(expm.directory, 'scvi'), exist_ok = False)

    # extract count matrix.
    import anndata
    counts = anndata.AnnData(
        X = atlas.layers[key_counts] \
            if key_counts is not None else atlas.X
    )

    counts.obs['batch'] = atlas.obs[key_batch].tolist() \
        if key_batch in atlas.obs.keys() else 'whole'
    counts.var_names = atlas.var_names.tolist()

    # extract metadata file.
    import scipy.sparse as sparse
    n_cells, n_genes = counts.X.shape
    metadata = anndata.AnnData(
        X = sparse.csr_matrix((n_cells, n_genes), dtype = np.float32),
        obs = atlas.obs, var = atlas.var
    )

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
    model.save(path)

    metadata.obsm[scvi_key] = scvi_pc
    metadata.uns['atlas'] = {
        'batch': 'batch',
        'latent': scvi_n_latent,
        'precomputed': scvi_key,
        'dimreduc': mde_key
    }

    atlas.obsm[scvi_key] = scvi_pc
    atlas.obs['atlas.batch'] = metadata.obs['batch']
    atlas.uns['atlas'] = {
        'batch': 'atlas.batch',
        'latent': scvi_n_latent,
        'precomputed': scvi_key,
        'dimreduc': mde_key
    }

    # build dimension reduction

    from exprmat.reduction.mde import mde
    import pymde

    mde_dim = mde(
        data = scvi_pc,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        embedding_dim = 2,
        constraint = pymde.Standardized(),
        repulsive_fraction = 0.7,
        verbose = False,
        n_neighbors = 15
    )

    metadata.obsm[mde_key] = mde
    atlas.obsm[mde_key] = mde
    metadata.write_h5ad(os.path.join(expm.directory, 'scvi', savename, 'metadata.h5ad'))
    pass


def is_atlas(expm: experiment):

    expm.check_merged('rna')

    if 'atlas' not in expm.mudata['rna'].uns.keys():
        error(f'do not contain an `atlas` key in uns slot. build atlas first.')
        return
    
    atlas = expm.mudata['rna'].uns['atlas']
    assert atlas['batch'] in expm.mudata['rna'].obs.keys()
    assert atlas['precomputed'] in expm.mudata['rna'].obsm.keys()
    assert atlas['dimreduc'] in expm.mudata['rna'].obsm.keys()


def project(
    atlas: experiment, 
    expm: experiment,
    
    counts_key = 'counts',
    batch_key = 'batch',
    key_query_latent = 'scvi',
    key_query_embeddings = 'mde',
    scvi_epoch_reduction = 3,
):
    """
    Query the reference atlas with a dataset
    
    Parameters
    ----------

    batch_key : str
        The observation metadata key specifying sample batches. This will be used to correct
        batch effect using ``scvi`` model. If not specified, the program will generate a obs
        slot named ``batch`` and assign all samples to the same batch. Note that if you have
        an observation metadata column named ``batch``, it will be overwritten.

    key_var : str
        The variable metadata key specifying the gene names. This should match the key selected
        in the atlas (by default, a list of ENSEMBL IDs). If not specified, the program will use
        the variable names. You should make sure that the contents in this column are unique.
        After the alignment, the variable names will be transformed to the same as the atlas.
        The original variable names will be stored in ``.index`` slot. You should keep a copy of
        that if you need them thereafter.
    
    key_query_latent : str
        The obsm key to store scVI latent space. If there is already a key with the same name,
        the calculation of scVI components will skip, and the data inside the slot will be used
        directly as the scVI latent space.
    
    key_query_embeddings : str
        The obsm key to store UMAP embeddings. This embeddings will *mostly* share the same 
        structure as the reference atlas. Since the exact UMAP model is used to transform the
        latent space. If ``retrain`` is set to ``False``, the UMAP will just serve as a prediction
        model to transform between dimensions without training on them. This is rather fast,
        but may introduce errors in the predicted embeddings (since the model have not seen
        the data totally during its training). Non-parametric model do not support retraining,
        and can only be used as a prediction model. 

        Parametric UMAP models have the capability to be retrained with new data. This will help
        the new data points better integrated into the atlas, and revealing more accurate
        alignment. However, the atlas embedding is somewhat affected by the new ones. Though we
        use landmarking points to help preserve the original structure, there may be some 
        small differences between the new atlas and the original one.

        If there is already a key with the same name, UMAP embedding calculation will be skipped.

    scvi_epoch_reduction : int
        Since the scVI model has been trained, we just need a few epochs to adapt it to the new
        data. The epochs may be less than what scVI expected to be. This saves a lot of time when
        running on CPU machines without reducing the performance too much. By default, the 
        reduction ratio is set to 4.
    
    n_jobs : int
        Number of threads to use when running UMAP embedding.

    Returns
    -------
    anndata.AnnData

        The modified anndata object. with the following slots set:
        
        * ``.obs``: ``batch``
        * ``.var``: ``index``, ``var_names``
        * ``.obsm``: ``key_query_latent``, ``key_query_embeddings``
        * ``.uns``: ``.align``

        These changes is made inplace, however, the modified object is still
        returned for convenience.
    """

    import scvi
    import anndata

    input = expm.mudata['rna']
    query = anndata.AnnData(
        X = input.X.copy() if counts_key is None 
        else input.layers[counts_key].copy()
    )

    query.obs_names = input.obs_names
    query.var_names = input.var_names
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
        

    if key_query_latent in input.obsm.keys():
        query.obsm[key_query_latent] = input.obsm[key_query_latent].copy()
    
    if key_query_embeddings in input.obsm.keys():
        query.obsm[key_query_embeddings] = input.obsm[key_query_embeddings].copy()


    if key_query_latent not in query.obsm.keys():
        
        info(f'preparing query data ...')
        scvi_model = os.path.join(
            atlas.directory, 'scvi', 
            atlas.subset if atlas.subset is not None else 'integrated'
        )

        scvi.model.SCVI.prepare_query_anndata(query, scvi_model)
        info(f'constructing query model for batch correction ...')
        query_model = scvi.model.SCVI.load_query_data(query, scvi_model)
    
        max_epochs_scvi = np.min([round((20000 / query.n_obs) * 400), 400]) // scvi_epoch_reduction
        info(f'will automatically train {max_epochs_scvi} epochs ...')
        query_model.train(max_epochs = int(max_epochs_scvi), plan_kwargs = { 'weight_decay': 0.0 })
        query.obsm[key_query_latent] = query_model.get_latent_representation()

    else: info(f'skipped calculation of scvi, since it already exist.')
    query_latent = query.obsm[key_query_latent]
    input.obsm[key_query_latent] = query_latent
    
    if key_query_embeddings not in query.obsm.keys():
        
        import pymde
        train_embedding = atlas.mudata['rna'].obsm[
            atlas.mudata['rna'].uns['atlas']['dimreduc']
        ]

        n_train = train_embedding.shape[0]
        train_data = atlas.mudata['rna'].obsm[
            atlas.mudata['rna'].uns['atlas']['precomputed']
        ]

        updated_embedding = pymde.preserve_neighbors(
            torch.vstack([train_data, query_latent]),
            constraint = pymde.Anchored(torch.arange(n_train), train_embedding),
            verbose = False,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            embedding_dim = 2,
            repulsive_fraction = 0.7,
            n_neighbors = 15
        ).embed(verbose = False)
        
        query.obsm[key_query_embeddings] = updated_embedding
        input.obsm[key_query_embeddings] = updated_embedding

    # expm.uns['atlas'] = {
    #     'batch': batch_key,
    #     'latent': query_latent.shape[1],
    #     'precomputed': key_query_latent,
    #     'dimreduc': key_query_embeddings
    # }

    return query