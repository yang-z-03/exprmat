
import numpy as np
import pandas as pd
from typing import Literal, Optional, Union
from scipy.sparse import spmatrix
import torch

from exprmat.ansi import warning

def mde(
    data, device: Optional[Literal['cpu', 'cuda']] = None, **kwargs,
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
    emb = pymde.preserve_neighbors(data, **default_args).embed(verbose = default_args['verbose'])

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()

    return emb