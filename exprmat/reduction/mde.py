
import numpy as np
import pandas as pd
from typing import Literal, Optional
import torch

from exprmat.ansi import warning


def mde(
    data, device: Optional[Literal['cpu', 'cuda']] = None, 
    n_dims = 2, repulsive_fraction = 0.7, verbose = False,
    n_neighbors = 15, eps: float = 0.00001, max_iter: int = 300, memory_size: int = 10,
    constraint = None, random_seed = 42,
    **kwargs,
) -> np.ndarray:
    '''
    This function is included in scvi-tools to provide an alternative to UMAP/TSNE that is GPU-
    accelerated. The appropriateness of use of visualization of high-dimensional spaces in single-
    cell omics remains an open research question.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, spmatrix, torch.Tensor]
        The source data to embed.

    device : Optional[Literal['cpu', 'cuda']] = None
        Whether to run on cpu or gpu ('cuda'). If None, tries to run on gpu if available.

    kwargs : dict
        Keyword args to :func:`pymde.preserve_neighbors`
    '''

    try: import pymde
    except: 
        warning('mde embedding require installation of package `pymde`.')
        return None
        
    if isinstance(data, pd.DataFrame):
        data = data.values

    if device is None:
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    default_args = {
        'embedding_dim': n_dims,
        'constraint': constraint,
        'repulsive_fraction': repulsive_fraction,
        'verbose': verbose,
        'device': device,
        'n_neighbors': n_neighbors,
    }

    problem = pymde.preserve_neighbors(data, **default_args)

    torch.random.manual_seed(random_seed)
    emb = problem.embed(verbose = verbose, eps = eps, max_iter = max_iter, memory_size = memory_size)

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()

    return emb


def mde_fit(
    data, fit, fit_mask, device = None, 
    eps: float = 0.00001, max_iter: int = 300, memory_size: int = 10,
    verbose = False,
    **kwargs
):
    
    try: import pymde
    except: 
        warning('mde embedding require installation of package `pymde`.')
        return None
    
    fitting = fit[fit_mask, :]
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    default_args = {
        'embedding_dim': 2,
        'constraint': pymde.Standardized(),
        'repulsive_fraction': 0.7,
        'verbose': False,
        'device': device,
        'n_neighbors': 15,
    }

    default_args.update(kwargs)
    emb = pymde.preserve_neighbors(
        torch.tensor(data, dtype = torch.float),
        constraint = pymde.Anchored(
            torch.nonzero(torch.tensor(fit_mask)).squeeze(), 
            torch.tensor(fitting, dtype = torch.float)
        ),
        verbose = verbose
    ).embed(
        verbose = verbose,
        eps = eps, max_iter = max_iter, 
        memory_size = memory_size
    )

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    
    return emb