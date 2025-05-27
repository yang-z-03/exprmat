'''
Summary accepts two to three categoricals in the obs metadata to reform the dataset
with summarizing statistics. Aggregates takes one obs category and one var category
to aggregate observations and variables.
'''

import numpy as np
import anndata as ad

from exprmat.utils import choose_layer


def aggregate(
    adata, data = 'X', method = 'mean', method_args = {},
    obs_key = 'sample', var_key = None
):
    source = choose_layer(adata, layer = data)

    # TODO: maybe some alternative method rather than silly iteration :(

    if (obs_key is None) and (var_key is not None):
        return aggregate_var(
            adata, data = data, method = method, method_args = method_args,
            var_key = var_key
        )
    
    if (var_key is None) and (obs_key is not None):
        return aggregate_obs(
            adata, data = data, method = method, method_args = method_args,
            obs_key = obs_key
        )
    
    obs_val = adata.obs_names.tolist()
    obs = adata.obs_names.to_numpy()
    if obs_key is not None:
        obs_val = adata.obs[obs_key].unique().tolist()
        obs = adata.obs[obs_key].to_numpy()
    
    var_val = adata.var_names.tolist()
    var = adata.var_names.to_numpy()
    if var_key is not None:
        var_val = adata.var[var_key].unique().tolist()
        var = adata.var[var_key].to_numpy()
    
    mat = np.ndarray(shape = (len(obs_val), len(var_val)), dtype = np.float32)
    meth = None

    methods = {
        'mean': np.mean
    }

    for i in range(len(obs_val)):
        for j in range(len(var_val)):
            
            mat[i, j] = methods[method](
                source[obs == obs_val[i], var == var_val[j]], 
                **method_args
            )
    
    annd = ad.AnnData(X = mat)
    annd.obs_names = obs_val
    annd.var_names = var_val
    return annd


def aggregate_obs(
    adata, data = 'X', method = 'mean', method_args = {}, obs_key = 'sample'
):
    
    source = choose_layer(adata, layer = data)

    obs_val = adata.obs_names.tolist()
    obs = adata.obs_names.to_numpy()
    if obs_key is not None:
        obs_val = adata.obs[obs_key].unique().tolist()
        obs = adata.obs[obs_key].to_numpy()
    
    mat = np.ndarray(shape = (len(obs_val), adata.n_vars), dtype = np.float32)

    methods = {
        'mean': np.mean
    }

    for i in range(len(obs_val)):
        mat[i, :] = methods[method](
            source[obs == obs_val[i], :],
            axis = 0,
            **method_args
        )
    
    annd = ad.AnnData(X = mat)
    annd.obs_names = obs_val
    annd.var_names = adata.var_names.tolist()
    annd.var = adata.var.copy()
    return annd


def aggregate_var(
    adata, data = 'X', method = 'mean', method_args = {}, var_key = 'module'
):
    
    source = choose_layer(adata, layer = data)

    var_val = adata.var_names.tolist()
    var = adata.var_names.to_numpy()
    if var_key is not None:
        var_val = adata.var[var_key].unique().tolist()
        var = adata.var[var_key].to_numpy()
    
    mat = np.ndarray(shape = (adata.n_obs, len(var_val)), dtype = np.float32)

    methods = {
        'mean': np.mean
    }

    for i in range(len(var_val)):
        mat[:, i] = methods[method](
            source[:, var == var_val[i]],
            axis = 1, # row sums
            **method_args
        )
    
    annd = ad.AnnData(X = mat)
    annd.obs_names = adata.obs_names.tolist()
    annd.obs = adata.obs.copy()
    annd.var_names = var_val
    return annd