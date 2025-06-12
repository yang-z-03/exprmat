
import warnings
import numpy as np
from scipy.sparse import csr_matrix, issparse
from exprmat.dynamics.utils import get_connectivities
from exprmat.ansi import error, warning, info


def not_yet_normalized(X):
    return np.allclose(np.ravel(X[:5].data if issparse(X) else X[:5]) % 1, 0, atol=1e-3)


def moments(
    adata,
    mode = "connectivities",
    neighbor_key = 'neighbors'
):

    layers = [layer for layer in {"spliced", "unspliced"} if layer in adata.layers]
    if any([not_yet_normalized(adata.layers[layer]) for layer in layers]):
        error('some layers of `spliced` and `unspliced` do not seems to be log normalized.')

    if "spliced" not in adata.layers.keys() or "unspliced" not in adata.layers.keys():
        warning("skipping moments, because un/spliced counts were not found.")
    else:
        info(f"computing moments based on {mode}")
        connectivities = get_connectivities(adata, neighbor_key = neighbor_key, mode = mode)

        adata.layers["ms"] = (
            csr_matrix.dot(connectivities, csr_matrix(adata.layers["spliced"]))
            .astype(np.float32).toarray()
        )

        adata.layers["mu"] = (
            csr_matrix.dot(connectivities, csr_matrix(adata.layers["unspliced"]))
            .astype(np.float32).toarray()
        )
        
    return adata


def second_order_moments(adata, neighbor_key = 'neighbors', mode = "connectivities", adjusted = False):

    connectivities = get_connectivities(adata, neighbor_key = neighbor_key, mode = mode)
    s, u = csr_matrix(adata.layers["spliced"]), csr_matrix(adata.layers["unspliced"])
    if s.shape[0] == 1:
        s, u = s.T, u.T

    Mss = csr_matrix.dot(connectivities, s.multiply(s)).astype(np.float32).toarray()
    Mus = csr_matrix.dot(connectivities, s.multiply(u)).astype(np.float32).toarray()
    
    if adjusted:
        Mss = 2 * Mss - adata.layers["ms"].reshape(Mss.shape)
        Mus = 2 * Mus - adata.layers["mu"].reshape(Mus.shape)
    
    return Mss, Mus


def second_order_moments_u(adata, neighbor_key = 'neighbors', mode = "connectivities"):
    
    connectivities = get_connectivities(adata, neighbor_key = neighbor_key, mode = mode)
    u = csr_matrix(adata.layers["unspliced"])
    Muu = csr_matrix.dot(connectivities, u.multiply(u)).astype(np.float32).toarray()
    return Muu


def get_moments(
    adata, layer = None, second_order = None, centered = True, 
    neighbor_key = 'neighbors', mode = "connectivities"
):
    
    connectivities = get_connectivities(adata, neighbor_key = neighbor_key, mode = mode)
    
    X = (
        adata.X
        if layer is None
        else adata.layers[layer]
        if isinstance(layer, str)
        else layer
    )

    X = (
        csr_matrix(X)
        if isinstance(layer, str) and layer in {"spliced", "unspliced"}
        else np.array(X)
        if not issparse(X)
        else X
    )

    if not issparse(X): X = X[:, ~ np.isnan(X.sum(0))]
    if second_order:
        X2 = X.multiply(X) if issparse(X) else X ** 2
        Mx = (
            csr_matrix.dot(connectivities, X2)
            if second_order
            else csr_matrix.dot(connectivities, X)
        )

        if centered:
            mu = csr_matrix.dot(connectivities, X)
            mu2 = mu.multiply(mu) if issparse(mu) else mu ** 2
            Mx = Mx - mu2
    
    else: Mx = csr_matrix.dot(connectivities, X)
    if issparse(X): Mx = Mx.astype(np.float32).toarray()
    return Mx
