
import os
import numpy as np
import torch

from exprmat.ansi import error, warning, info
from exprmat.reader.experiment import experiment


def construct_atlas(
    adata,
    expm_dir,
    expm_subset,
    key_counts = 'counts',
    key_batch = 'batch',
    hvg = 'hvg',
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

    scvi_kwargs = {},
    scanvi_kwargs = {},
    scanvi_train_kwargs = {},
    scvi_train_kwargs = {},

    use_model = None
):
    """
    构建单细胞数据集的参考图谱（atlas）
    
    该函数使用 scVI 和 scANVI 模型构建单细胞转录组数据的参考图谱，
    实现批次效应校正和细胞类型注释。函数会训练 scVI 模型进行批次校正，
    然后使用 scANVI 模型进行半监督学习，将已知细胞类型信息整合到潜在表示中。
    
    Parameters
    ----------
    expm : experiment
        实验对象，包含需要构建图谱的单细胞数据
    key_counts : str, optional
        计数矩阵在 layers 中的键名，默认为 'counts'。
        如果为 None，则使用 .X 矩阵
    key_batch : str, optional
        批次信息在 obs 中的列名，默认为 'batch'。
        如果不存在，所有细胞将被标记为 'whole'
    batch_cell_filter : int, optional
        批次中细胞数量的最小阈值，默认为 50。
        低于此阈值的批次将被标记为 'outliers' 并从分析中移除
    scvi_n_epoch : int, optional
        scVI 模型训练的最大轮数，默认为 None（自动计算）。
        自动计算基于细胞数量：min(round((20000/n_cells) * 400), 400)
    scvi_n_latent : int, optional
        scVI 模型潜在空间维度，默认为 10
    scvi_n_hidden : int, optional
        scVI 模型隐藏层神经元数量，默认为 128
    scvi_n_layers : int, optional
        scVI 模型隐藏层数量，默认为 1
    scvi_dropout_rate : float, optional
        scVI 模型 dropout 率，默认为 0.1
    scvi_dispersion : str, optional
        scVI 模型离散度参数，默认为 'gene'
    scvi_gene_likelihood : str, optional
        scVI 模型基因似然分布，默认为 'zinb'（零膨胀负二项分布）
    scvi_latent_distrib : str, optional
        scVI 模型潜在分布类型，默认为 'normal'
    scvi_key : str, optional
        scVI 潜在表示在 obsm 中的键名，默认为 'scvi'
    annotation : str, optional
        细胞类型注释在 obs 中的列名，默认为 'cell.type'
    scanvi_key : str, optional
        scANVI 潜在表示在 obsm 中的键名，默认为 'scanvi'
    scanvi_unlabel : str, optional
        未标记细胞的类别名称，默认为 'unknown'
    scanvi_max_epochs : int, optional
        scANVI 模型训练的最大轮数，默认为 30
    scanvi_samples_per_label : int, optional
        每个标签的采样数量，默认为 200
        
    Returns
    -------
    None
        函数不返回值，但会：
        1. 在 expm.mudata['rna'] 中添加 scVI 和 scANVI 的潜在表示
        2. 在 expm.directory/scvi/ 目录下保存训练好的模型
        3. 在 expm.mudata['rna'].uns 中添加图谱元数据
        
    Notes
    -----
    - 函数会自动创建 scvi 目录结构
    - 如果目标路径已存在，函数会报错并退出
    - 训练轮数会根据数据规模自动调整
    - 小批次样本会被标记为 'outliers' 并从分析中移除
    - 模型会保存到 expm.directory/scvi/{savename}/ 目录下
    - 同时会生成一个轻量级的元数据文件 metadata.h5ad
    
    Examples
    --------
    >>> from exprmat.transfer.atlas import construct_atlas
    >>> # 构建参考图谱
    >>> construct_atlas(
    ...     expm, 
    ...     scvi_n_latent = 20, 
    ...     scvi_n_hidden = 256,
    ...     annotation = 'cell.type'
    ... )
    """
    
    import warnings
    warnings.filterwarnings('ignore')
    from exprmat.utils import supports_tensorcore
    if supports_tensorcore():
        torch.set_float32_matmul_precision('high')

    atlas = adata
    savename = expm_subset if expm_subset is not None else 'integrated'
    path = os.path.join(expm_dir, 'scvi', savename)
    scanvipath = os.path.join(expm_dir, 'scanvi', savename)
    
    if os.path.exists(path):
        error('the destination path `scvi/{savename}` is already occupied.')
    else: os.makedirs(os.path.join(expm_dir, 'scvi'), exist_ok = True)


    # extract count matrix.
    import anndata
    counts = anndata.AnnData(
        X = atlas[:, atlas.var[hvg]].layers[key_counts].copy() \
            if key_counts is not None else atlas[:, atlas.var[hvg]].X.copy()
    )

    counts.obs['batch'] = atlas.obs[key_batch].tolist() \
        if key_batch in atlas.obs.keys() else 'whole'
    counts.obs['annotation'] = atlas.obs[annotation].tolist()
    counts.var_names = atlas.var_names[atlas.var[hvg] == True].tolist()
    info(f'prepare count matrix of size {counts.n_obs} * {counts.n_vars}')


    # build scvi model
    import scvi
    import scipy.sparse as sparse
    
    # extract metadata file.
    n_cells, n_genes = counts.X.shape
    metadata = anndata.AnnData(
        X = sparse.csr_matrix((n_cells, n_genes), dtype = np.float32),
        obs = counts.obs, var = counts.var
    )
    
    if use_model:
        model = scvi.model.SCVI.load(os.path.join(expm_dir, 'scvi', use_model), adata = counts)
        scvi_pc = adata.obsm[use_model]
        scvi_n_latent = adata.uns[use_model]['n_latent']
        if adata.uns[use_model]['batch_key'] != key_batch:
            prev_batch = adata.uns[use_model]['batch_key']
            error(f'failed to re-use trained scvi model since it was trained on batch `{prev_batch}`.')
        model.save(path)

    else:
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
            latent_distribution = scvi_latent_distrib,
            **scvi_kwargs
        )

        max_epochs_scvi = np.min([round((20000 / counts.n_obs) * 400), 400]) \
            if scvi_n_epoch is None else scvi_n_epoch
        info(f'will train {max_epochs_scvi} epochs.')
        model.train(max_epochs = int(max_epochs_scvi), **scvi_train_kwargs)
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
        'hvg': None
    }

    atlas.obsm[scvi_key] = scvi_pc
    atlas.obs['atlas.batch'] = metadata.obs['batch'].tolist()
    atlas.uns['atlas'] = {
        'batch': 'atlas.batch',
        'annotation': annotation,
        'latent': scvi_n_latent,
        'scvi': scvi_key,
        'scanvi': scanvi_key,
        'hvg': hvg
    }

    
    # build scanvi model
    info(f'training scanvi model on label `{annotation}`')
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        unlabeled_category = scanvi_unlabel,
        labels_key = 'annotation',
        **scanvi_kwargs
    )

    scanvi_model.train(
        max_epochs = scanvi_max_epochs, 
        n_samples_per_label = scanvi_samples_per_label,
        **scanvi_train_kwargs
    )

    atlas.obsm[scanvi_key] = scanvi_model.get_latent_representation()
    metadata.obsm[scanvi_key] = atlas.obsm[scanvi_key]

    info(f'trained scanvi model saved.')
    scanvi_model.save(scanvipath)
    info(f'saving metadata of atlas as a light version.')
    metadata.write_h5ad(os.path.join(expm_dir, 'scvi', savename, 'metadata.h5ad'))
    warnings.filterwarnings('default')


def is_atlas(adata):

    if 'atlas' not in adata.uns.keys():
        warning(f'do not contain an `atlas` key in uns slot. build atlas first.')
        return False
    
    atlas = adata.uns['atlas']
    if atlas['batch'] not in adata.obs.keys():
        warning(f'`{atlas["batch"]}` do not exist.')
        return False
    
    if atlas['annotation'] not in adata.obs.keys():
        warning(f'`{atlas["annotation"]}` do not exist.')
        return False

    if atlas['scvi'] not in adata.obsm.keys():
        warning(f'`{atlas["scvi"]}` do not exist.')
        return False

    if atlas['scanvi'] not in adata.obsm.keys():
        warning(f'`{atlas["scanvi"]}` do not exist.')
        return False

    return True


def embed(
    atlas: experiment, 
    expm,
    
    counts_key = 'counts',
    batch_key = 'batch',
    key_query_latent = 'scanvi',
    scanvi_unlabel = 'unknown',
    scvi_epoch_reduction = 3,
    predict_labels = 'annot',

    use_scanvi = True
):
    
    from exprmat.utils import supports_tensorcore
    if supports_tensorcore():
        torch.set_float32_matmul_precision('high')

    import scvi
    import anndata

    input = expm
    query = anndata.AnnData(
        X = input.X.copy() if counts_key is None 
        else input.layers[counts_key].copy()
    )

    query.obs_names = input.obs_names
    query.var_names = input.var_names
    query.obs['annotation'] = scanvi_unlabel
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
        

    if use_scanvi:
        info(f'preparing query data ...')
        scvi_path = os.path.join(
            atlas.directory, 'scanvi', 
            atlas.subset if atlas.subset is not None else 'integrated'
        )

        scvi.model.SCANVI.prepare_query_anndata(query, scvi_path)
        info(f'constructing query model for batch correction ...')
        query_model = scvi.model.SCANVI.load_query_data(query, scvi_path)

        max_epochs_scvi = np.min([round((20000 / query.n_obs) * 400), 400]) // scvi_epoch_reduction
        info(f'will automatically train {max_epochs_scvi} epochs ...')
        query_model.train(max_epochs = int(max_epochs_scvi), plan_kwargs = { 'weight_decay': 0.0 })

        info(f'getting latent representations ...')
        query.obsm[key_query_latent] = query_model.get_latent_representation()
        input.obsm[key_query_latent] = query.obsm[key_query_latent]

        if predict_labels:
            query.obs[predict_labels] = query_model.predict()
            input.obs[predict_labels] = query.obs[predict_labels]
            input.obs[predict_labels] = input.obs[predict_labels].astype('category')
            query.obs[predict_labels] = query.obs[predict_labels].astype('category')
        
    else:
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

        if predict_labels:
            warning('`predict_labels` will be ignored when setting `use_scanvi` to false.')
    
    return query


def project(
    atlas: experiment, 
    expm,
    
    atlas_modality = 'rna',
    atlas_embedding = 'umap',
    counts_key = 'counts',
    batch_key = 'batch',
    key_query_latent = 'scanvi',
    scanvi_unlabel = 'unknown',
    scvi_epoch_reduction = 3,
    predict_labels = 'annot',
    key_added = 'projection',
    embed_decay = 30,

    re_embed = False,
    use_scanvi = True,
    **kwargs
):
    
    data_meta = atlas[atlas_modality].obs[[
        atlas[atlas_modality].uns['atlas']['batch'],
        atlas[atlas_modality].uns['atlas']['annotation']
    ]].copy()
    data_meta.columns = ['batch', 'annotation']

    embed(
        atlas = atlas,
        expm = expm,
        counts_key = counts_key,
        batch_key = batch_key,
        key_query_latent = key_query_latent,
        scanvi_unlabel = scanvi_unlabel,
        scvi_epoch_reduction = scvi_epoch_reduction,
        predict_labels = predict_labels,
        use_scanvi = use_scanvi
    )

    # copy metadata from atlas to expm.
    
    atlas.check_merged(atlas_modality)
    atlas = atlas[atlas_modality]
    expm = expm

    data = atlas.obsm[atlas_embedding]

    data_meta['x'] = data[:, 0]
    data_meta['y'] = data[:, 1]

    import pandas as pd
    if isinstance(data, pd.DataFrame): data = data.values
    if not isinstance(data, np.ndarray): data = np.array(data)
    assert data.shape[1] == 2

    key_atlas = 'scanvi' if use_scanvi else 'scvi'

    if not re_embed:
        n_cells = expm.n_obs
        n_ref = atlas.n_obs // embed_decay
        if atlas.n_obs < min(100, embed_decay):
            warning('the atlas subsample to embed is too small (<100 or embed_decay)')
            warning('you should set embed_decay to 1 when embedding to a very small atlas.')

        scanvi_atlas = atlas.obsm[atlas.uns['atlas'][key_atlas]]
        scanvi_expm = expm.obsm[key_query_latent]

        if embed_decay != 1:
            embed_mask = np.random.choice([x for x in range(atlas.n_obs)], n_ref, False)
            scanvi_atlas = scanvi_atlas[embed_mask, :].copy()
            data = data[embed_mask, :].copy()

        merged_latent = np.concat([scanvi_atlas, scanvi_expm], axis = 0)
        merged_embed = np.concat([data, np.zeros(shape = (n_cells, 2))], axis = 0)
        mask = np.array([True] * n_ref + [False] * n_cells)

        from exprmat.reduction.mde import mde_fit, mde
        emb = mde_fit(merged_latent, merged_embed, mask, **kwargs)

        expm.obsm[key_added] = emb[~mask, :]
        expm.uns[key_added] = data_meta
    
    else:
        n_cells = expm.n_obs
        n_ref = atlas.n_obs

        scanvi_atlas = atlas.obsm[atlas.uns['atlas'][key_atlas]]
        scanvi_expm = expm.obsm[key_query_latent]

        merged_latent = np.concat([scanvi_atlas, scanvi_expm], axis = 0)
        merged_embed = np.concat([data, np.zeros(shape = (n_cells, 2))], axis = 0)
        mask = np.array([True] * n_ref + [False] * n_cells)

        from exprmat.reduction.mde import mde_fit, mde
        emb = mde(merged_latent, **kwargs)

        expm.obsm[key_added] = emb[~mask, :]
        data_meta['x'] = emb[mask, 0]
        data_meta['y'] = emb[mask, 1]
        expm.uns[key_added] = data_meta
