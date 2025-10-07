
import os
import numpy as np
import torch

from exprmat.ansi import error, warning, info
from exprmat.reader.experiment import experiment


def construct_atlas(
    expm: experiment, 
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
    scvi_train_kwargs = {}
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
    torch.set_float32_matmul_precision('high')

    atlas = expm.mudata['rna']
    savename = expm.subset if expm.subset is not None else 'integrated'
    path = os.path.join(expm.directory, 'scvi', savename)
    scanvipath = os.path.join(expm.directory, 'scanvi', savename)
    
    if os.path.exists(path):
        error('the destination path `scvi/{savename}` is already occupied.')
    else: os.makedirs(os.path.join(expm.directory, 'scvi'), exist_ok = True)


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
    key_query_latent = 'scanvi',
    scanvi_unlabel = 'unknown',
    scvi_epoch_reduction = 3,
    predict_labels = 'annot'
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
    
    return query