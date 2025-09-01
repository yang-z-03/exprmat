

def run_diffmap(
    adata, *, neighbors_key = None, key_added = 'diffmap', 
    n_comps = 3, sort = 'decrease', density_normalize = True,
    use_gpu = False
):

    if use_gpu:

        import cupy as cp
        from cupyx.scipy import sparse
        from cupyx.scipy.sparse import linalg
        from scipy.sparse import issparse

        if neighbors_key:
            connectivities = adata.obsp[adata.uns[neighbors_key]['connectivities_key']]
        else: connectivities = adata.obsp["connectivities"]
        
        if issparse(connectivities):
            W = sparse.csr_matrix(connectivities, dtype = cp.float32)
        else: W = cp.asarray(connectivities)

        if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = cp.asarray(W.sum(axis=0))
            if not sparse.issparse(W): Q = cp.diag(1.0 / q)
            else: Q = sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q

        else: K = W

        # z[i] is the square root of the row sum of K 
        z = cp.sqrt(cp.asarray(K.sum(axis=0)))

        if not sparse.issparse(K): Z = cp.diag(1.0 / z)
        else: Z = sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        matrix = Z @ K @ Z

        if n_comps == 0: evals, evecs = linalg.eigsh(matrix)
        
        else:
            n_comps = min(matrix.shape[0] - 1, n_comps)
            # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
            ncv = None
            which = "LM" if sort == "decrease" else "SM"
            # it pays off to increase the stability with a bit more precision
            matrix = matrix.astype(cp.float64)
            evals, evecs = linalg.eigsh(matrix, k=n_comps, which=which, ncv=ncv)
            evals, evecs = evals.astype(cp.float32), evecs.astype(cp.float32)
        
        if sort == "decrease":
            evals = evals[::-1]
            evecs = evecs[:, ::-1]
        
        adata.uns[key_added] = evals.get()
        adata.obsm[key_added] = evecs.get()
        del W, Q, Z, K, evals, evecs, matrix, z
    
    else:

        import scanpy as sc
        sc.tl.diffmap(
            adata, n_comps = n_comps,
            neighbors_key = neighbors_key
        )

        adata.obsm[key_added] = adata.obsm['X_diffmap']
        del adata.obsm['X_diffmap']

        adata.uns[key_added] = adata.uns['diffmap_evals']
        del adata.uns['diffmap_evals']