
from typing import Literal
import scvi as scvitools


def scvi(
    adata, batch_key = 'batch', layer_key = 'X',

    # model arguments
    n_hidden: int = 128,
    n_latent: int = 30,
    n_layers: int = 3,
    dropout_rate: float = 0.1,
    dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = "gene",
    gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = "zinb",
    use_observed_lib_size: bool = True,
    latent_distribution: Literal['normal', 'ln'] = "normal",

    # trainer arguments
    max_epochs: int | None = None, 
    accelerator: str = "auto", 
    devices: int | list[int] | str = "auto", 
    train_size: float | None = None, 
    validation_size: float | None = None, 
    shuffle_set_split: bool = True, 
    load_sparse_tensor: bool = False, 
    batch_size: int = 128, 
    early_stopping: bool = False, 
    datasplitter_kwargs: dict | None = None, 
    plan_kwargs: dict | None = None, 
    datamodule = None,
    enable_progress_bar = True, 
    progress_bar_refresh_rate = 1, 
    simple_progress_bar = True, 
    logger = None, 
    log_every_n_steps = 10,

    # seeding options
    seeding = False,
    label_key = 'cell.type',
    unknown_value = 'imputed',
    scanvi_training_epochs = None,

    scvi_train_kwargs = {}
):

    # apply inplace with adata.X.
    # so, you should manage a new data file first if you do not want side effects.

    import torch
    from exprmat.utils import supports_tensorcore
    if supports_tensorcore():
        torch.set_float32_matmul_precision('high')

    scvitools.model.SCVI.setup_anndata(
        adata, layer = None if layer_key == 'X' else layer_key, 
        batch_key = batch_key, labels_key = label_key if seeding else None
    )

    model = scvitools.model.SCVI(
        adata, n_layers = n_layers, n_latent = n_latent, n_hidden = n_hidden,
        dropout_rate = dropout_rate, dispersion = dispersion,
        gene_likelihood = gene_likelihood, use_observed_lib_size = use_observed_lib_size,
        latent_distribution = latent_distribution
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.train(
            max_epochs = max_epochs, 
            accelerator = accelerator, 
            devices = devices, 
            train_size = train_size, 
            validation_size = validation_size, 
            shuffle_set_split = shuffle_set_split, 
            load_sparse_tensor = load_sparse_tensor, 
            batch_size = batch_size, 
            early_stopping = early_stopping, 
            datasplitter_kwargs = datasplitter_kwargs, 
            plan_kwargs = plan_kwargs, 
            datamodule = datamodule,
            enable_progress_bar = enable_progress_bar, 
            progress_bar_refresh_rate = progress_bar_refresh_rate, 
            simple_progress_bar = simple_progress_bar, 
            logger = logger, 
            log_every_n_steps = log_every_n_steps,
            ** scvi_train_kwargs
        )

    if seeding:

        scanvi_model = scvitools.model.SCANVI.from_scvi_model(
            scvi_model = model,
            unlabeled_category = unknown_value,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scanvi_model.train(
                max_epochs = scanvi_training_epochs, 
                accelerator = accelerator, 
                devices = devices
            )

        model = scanvi_model


    latent = model.get_latent_representation()
    if seeding:
        imputed = model.predict(adata)
        return model, latent, imputed
    else: return model, latent