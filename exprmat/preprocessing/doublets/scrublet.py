import numpy as np
import scipy
import scipy.sparse
import anndata as ad

from exprmat.ansi import warning, info
from exprmat.preprocessing import (
    normalize, highly_variable, log_transform,
    index_to_bool, bool_to_index, scale)
from exprmat.preprocessing.sparse import sparse_var
from exprmat.reduction.pca import pca
from exprmat.reduction.nn import knn_graph


def subsample_counts(E, rate, original_totals, random_seed = 0):
    if rate < 1:
        np.random.seed(random_seed)
        E.data = np.random.binomial(np.round(E.data).astype(int), rate)
        current_totals = E.sum(1).A.squeeze()
        unsampled_orig_totals = original_totals - current_totals
        unsampled_downsamp_totals = np.random.binomial(np.round(unsampled_orig_totals).astype(int), rate)
        final_downsamp_totals = current_totals + unsampled_downsamp_totals
    else: final_downsamp_totals = original_totals
    return E, final_downsamp_totals


def scrublet_init(
    adata, layer_count = 'X', 
    sim_doublet_ratio = 2.0, n_neighbors = None, 
    expected_doublet_rate = 0.1, stdev_doublet_rate = 0.02, 
    random_state = 0
):
    '''
    Initialize Scrublet object with counts matrix and doublet prediction parameters

    Parameters
    ----------

    counts_matrix : scipy sparse matrix or ndarray, shape (n_cells, n_genes)
        Matrix containing raw (unnormalized) UMI-based transcript counts. Converted 
        into a scipy.sparse.csc_matrix.

    total_counts : ndarray, shape (n_cells,), optional (default: None)
        Array of total UMI counts per cell. If `None`, this is calculated as the 
        row sums of `counts_matrix`. 

    sim_doublet_ratio : float, optional (default: 2.0)
        Number of doublets to simulate relative to the number of observed transcriptomes.

    n_neighbors : int, optional (default: None)
        Number of neighbors used to construct the KNN graph of observed transcriptomes 
        and simulated doublets. If `None`, this is set to round(0.5 * sqrt(n_cells))

    expected_doublet_rate : float, optional (default: 0.1)
        The estimated doublet rate for the experiment.

    stdev_doublet_rate : float, optional (default: 0.02)
        Uncertainty in the expected doublet rate.

    random_state : int, optional (default: 0)
        Random state for doublet simulation, approximate nearest neighbor search, 
        and PCA / TruncatedSVD.

    '''
    
    # initialize counts matrices
    adata.uns['simulated'] = None
    expr = adata.X if layer_count == 'X' else adata.layers[layer_count]

    if 'n.umi' not in adata.obs.keys():
        adata.obs['n.umi'] = expr.sum(1).A.squeeze()

    # self._gene_filter = np.arange(self._E_obs.shape[1])
    # self._embeddings = {}

    adata.uns['scrublet'] = {
        'counts': layer_count,
        'sim_doublet_ratio': sim_doublet_ratio,
        'n_neighbors': n_neighbors,
        'expected_doublet_rate': expected_doublet_rate,
        'stdev_doublet_rate': stdev_doublet_rate,
        'random_state': random_state
    }

    if adata.uns['scrublet']['n_neighbors'] is None:
        adata.uns['scrublet']['n_neighbors'] = int(round(0.5 * np.sqrt(expr.shape[0])))
    
    return adata


def scrublet(
    adata, 
    synthetic_doublet_umi_subsampling = 1.0, 
    use_approx_neighbors = True, 
    distance_metric = 'euclidean', 
    get_doublet_neighbor_parents = False, 
    min_counts = 3, 
    min_cells = 3, 
    min_gene_variability_pct = 85,
    n_comp = 30, 
    svd_solver = 'arpack',
    verbose = True,
    force_recalculate = False
):
    '''
    Standard pipeline for preprocessing, doublet simulation, and doublet prediction.
    Automatically sets a threshold for calling doublets, but it's best to check 
    this by running plot_histogram() afterwards and adjusting threshold  with 
    call_doublets(threshold=new_threshold) if necessary.

    Parameters
    -----------

    synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, each doublet is 
        created by simply adding the UMIs from two randomly sampled observed transcriptomes. 
        For values less than 1, the UMI counts are added and then randomly sampled at the 
        specified rate.

    use_approx_neighbors : bool, optional (default: True)
        Use approximate nearest neighbor method (annoy) for the KNN 
        classifier.

    distance_metric : str, optional (default: 'euclidean')
        Distance metric used when finding nearest neighbors. For list of valid values, 
        see the documentation for annoy (if `use_approx_neighbors` is True) or 
        sklearn.neighbors.NearestNeighbors (if `use_approx_neighbors` is False).
        
    get_doublet_neighbor_parents : bool, optional (default: False)
        If True, return the parent transcriptomes that generated the doublet neighbors 
        of each observed transcriptome. This information can be used to infer the cell 
        states that generated a given doublet state.

    min_counts : float, optional (default: 3)
        Used for gene filtering prior to PCA. Genes expressed at fewer than `min_counts` 
        in fewer than `min_cells` (see below) are excluded.

    min_cells : int, optional (default: 3)
        Used for gene filtering prior to PCA. Genes expressed at fewer than `min_counts` 
        (see above) in fewer than `min_cells` are excluded.

    min_gene_variability_pctl : float, optional (default: 85.0)
        Used for gene filtering prior to PCA. Keep the most highly variable genes
        (in the top min_gene_variability_pctl percentile), as measured by 
        the v-statistic [Klein et al., Cell 2015].

    n_prin_comps : int, optional (default: 30)
        Number of principal components used to embed the transcriptomes prior to 
        k-nearest-neighbor graph construction.

    svd_solver : str, optional (default: 'arpack')
        SVD solver to use. See available options for `svd_solver` from 
        `sklearn.decomposition.PCA` or `algorithm` from `sklearn.decomposition.TruncatedSVD`

    verbose : bool, optional (default: True)
        If True, print progress updates.
    '''

    adata.uns['simulated'] = None

    if verbose: info('preprocessing observation count matrix ...')
    expr = adata.X if adata.uns['scrublet']['counts'] == 'X' else \
        adata.layers[adata.uns['scrublet']['counts']]
    
    if 'n.umi' not in adata.obs.keys():
        adata.obs['n.umi'] = expr.sum(1).A.squeeze()
    
    postnorm_total = adata.obs['n.umi'].mean()

    if 'norm' in adata.layers.keys(): 
        warning('[!] use existing slot `norm` as normalized linear.')
    else: normalize(
        adata, counts = adata.uns['scrublet']['counts'], dest = 'norm',
        method = 'total', target_total = postnorm_total, 
        total_counts = adata.obs['n.umi']
    )
    
    if 'ff.hvg' in adata.var.keys():
        warning('[!] use existing slot `ff.hvg` as gene filter') 
    else: highly_variable(
        adata, counts = adata.uns['scrublet']['counts'], dest = 'ff',
        method = 'ff', min_counts = min_counts, min_cells = min_cells, 
        min_vscore_pct = min_gene_variability_pct
    )
    
    f_counts = expr[bool_to_index(adata.obs['qc']), :][:, bool_to_index(adata.var['ff.hvg'])]
    subset = ad.AnnData(f_counts)
    subset.obs['n.umi'] = adata.obs['n.umi']
    adata.uns['hvg.subset'] = subset

    if verbose: info('simulating doublets ...')
    adata.uns['scrublet']['synthetic_doublet_umi_subsampling'] = \
        synthetic_doublet_umi_subsampling
    simulate_doublets(
        adata, f_counts, 
        sim_doublet_ratio = adata.uns['scrublet']['sim_doublet_ratio'], 
        synthetic_doublet_umi_subsampling = synthetic_doublet_umi_subsampling
    )

    normalize(
        adata.uns['hvg.subset'], counts = 'X', dest = 'norm.1e6',
        method = 'total', target_total = 1e6, 
        total_counts = adata.obs['n.umi']
    )

    normalize(
        adata.uns['simulated'], counts = 'X', dest = 'norm.1e6',
        method = 'total', target_total = 1e6, 
        total_counts = adata.uns['simulated'].obs['n.umi']
    )

    log_transform(adata.uns['hvg.subset'], norm = 'norm.1e6')
    log_transform(adata.uns['simulated'], norm = 'norm.1e6')

    gene_means = adata.uns['hvg.subset'].layers['lognorm'].mean(0)
    gene_stdevs = np.sqrt(sparse_var(adata.uns['hvg.subset'].layers['lognorm']))
    
    scale(
        adata.uns['hvg.subset'], lognorm = 'lognorm', dest = 'scaled', 
        gene_mean = gene_means, gene_stdev = gene_stdevs
    )

    scale(
        adata.uns['simulated'], lognorm = 'lognorm', dest = 'scaled', 
        gene_mean = gene_means, gene_stdev = gene_stdevs
    )

    if verbose: info('embedding using pca ...')
    emb, _, _, _, _, _, pca_embedder = \
        pca(adata.uns['hvg.subset'].layers['scaled'], n_comp = n_comp, svd_solver = svd_solver)
    emb_sim = pca_embedder.transform(adata.uns['simulated'].layers['scaled'])
    adata.uns['hvg.subset'].obsm['pca'] = emb
    adata.uns['simulated'].obsm['pca'] = emb_sim
    
    if verbose: info('calculating doublet scores ...')
    calculate_doublet_scores(
        adata,
        use_approx_neighbors = use_approx_neighbors,
        distance_metric = distance_metric
    )

    return call_doublets(adata, verbose = verbose)


def simulate_doublets(adata, e_obs, sim_doublet_ratio = None, synthetic_doublet_umi_subsampling = 1.0):
    '''
    Simulate doublets by adding the counts of random observed transcriptome pairs.

    Arguments
    ---------
    sim_doublet_ratio : float, optional (default: None)
        Number of doublets to simulate relative to the number of observed transcriptomes. 
        If `None`, self.sim_doublet_ratio is used.

    synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, each doublet is 
        created by simply adding the UMIs from two randomly sampled observed transcriptomes. 
        For values less than 1, the UMI counts are added and then randomly sampled at the 
        specified rate.
    '''

    if sim_doublet_ratio is None:
        sim_doublet_ratio = adata.uns['scrublet']['sim_doublet_ratio']
    else: adata.uns['scrublet']['sim_doublet_ratio'] = sim_doublet_ratio

    n_obs = e_obs.shape[0]
    n_sim = int(n_obs * sim_doublet_ratio)

    np.random.seed(adata.uns['scrublet']['random_state'])
    pair_index = np.random.randint(0, n_obs, size = (n_sim, 2))
    
    E1 = e_obs[pair_index[:, 0],:]
    E2 = e_obs[pair_index[:, 1],:]
    tots1 = adata.obs['n.umi'].iloc[pair_index[:,0]].to_numpy()
    tots2 = adata.obs['n.umi'].iloc[pair_index[:,1]].to_numpy()
    
    if synthetic_doublet_umi_subsampling < 1:
        simulated, simulated_n = subsample_counts(
            E1 + E2, synthetic_doublet_umi_subsampling, 
            tots1 + tots2, random_seed = adata.uns['scrublet']['random_state']
        )

    else:
        simulated = E1 + E2
        simulated_n = tots1 + tots2
    
    sim = ad.AnnData(simulated)
    sim.obs['n.umi'] = simulated_n
    sim.obsm['pairs'] = pair_index
    adata.uns['simulated'] = sim
    return


def calculate_doublet_scores(
    adata, 
    use_approx_neighbors = True, 
    distance_metric = 'euclidean'
):
    '''
    Calculate doublet scores for observed transcriptomes and simulated doublets.
    
    Arguments
    ---------

    use_approx_neighbors : bool, optional (default: True)
        Use approximate nearest neighbor method (annoy) for the KNN classifier.

    distance_metric : str, optional (default: 'euclidean')
        Distance metric used when finding nearest neighbors. For list of valid values, 
        see the documentation for annoy (if `use_approx_neighbors` is True) or 
        sklearn.neighbors.NearestNeighbors (if `use_approx_neighbors` is False).
        
    get_doublet_neighbor_parents : bool, optional (default: False)
        If True, return the parent transcriptomes that generated the doublet neighbors 
        of each observed transcriptome. This information can be used to infer the cell 
        states that generated a given doublet state.
    '''

    doublet_nn(
        adata,
        k = adata.uns['scrublet']['n_neighbors'],
        exp_doub_rate = adata.uns['scrublet']['expected_doublet_rate'],
        stdev_doub_rate = adata.uns['scrublet']['stdev_doublet_rate'],
        use_approx_nn = use_approx_neighbors, 
        distance_metric = distance_metric
    )


def doublet_nn(
    adata, k = 40, use_approx_nn = True, distance_metric = 'euclidean', 
    exp_doub_rate = 0.1, stdev_doub_rate = 0.03
):
    manifold = np.vstack((adata.uns['hvg.subset'].obsm['pca'], adata.uns['simulated'].obsm['pca']))
    doub_labels = np.concatenate((
        np.zeros(adata.uns['hvg.subset'].obsm['pca'].shape[0], dtype = int), 
        np.ones(adata.uns['simulated'].obsm['pca'].shape[0], dtype = int)
    ))

    n_obs = np.sum(doub_labels == 0)
    n_sim = np.sum(doub_labels == 1)

    # adjust k (number of nearest neighbors) based on the ratio of
    # simulated to observed cells
    k_adj = int(round(k * (1 + n_sim / float(n_obs))))

    # find k_adj nearest neighbors
    _, neighbors, _ = knn_graph(
        manifold, k = k_adj, dist_metric = distance_metric, 
        approx = use_approx_nn, random_seed = adata.uns['scrublet']['random_state']
    )

    # calculate doublet score based on ratio of simulated cell neighbors vs.
    # observed cell neighbors
    doub_neigh_mask = doub_labels[neighbors] == 1
    n_sim_neigh = doub_neigh_mask.sum(1)
    n_obs_neigh = doub_neigh_mask.shape[1] - n_sim_neigh

    rho = exp_doub_rate
    r = n_sim / float(n_obs)
    nd = n_sim_neigh.astype(float)
    ns = n_obs_neigh.astype(float)
    N = float(k_adj)

    # bayesian
    q = (nd + 1) / (N + 2)
    ld = q * rho / r / (1 - rho - q * (1 - rho - rho / r))
    se_q = np.sqrt(q * (1 - q) / (N + 3))
    se_rho = stdev_doub_rate
    se_ld = (
        q
        * rho
        / r
        / (1 - rho - q * (1 - rho - rho / r)) ** 2
        * np.sqrt((se_q / q * (1 - rho)) ** 2 + (se_rho / rho * (1 - q)) ** 2)
    )

    adata.obs['score.doublet'] = ld[doub_labels == 0]
    adata.uns['simulated'].obs['score.doublet'] = ld[doub_labels == 1]
    adata.obs['score.doublet.se'] = se_ld[doub_labels == 0]
    adata.uns['simulated'].obs['score.doublet.se'] = se_ld[doub_labels == 1]
    return

    
def call_doublets(adata, threshold = None, verbose = True):
    '''
    Call trancriptomes as doublets or singlets

    Parameters
    -----------

    threshold : float, optional (default: None) 
        Doublet score threshold for calling a transcriptome a doublet. If `None`, this is 
        set automatically by looking for the minimum between the two modes of the 
        `uns['simulated'].obs['score.doublet']` histogram. It is best practice to check 
        the threshold visually using the histogram and/or based on co-localization of 
        predicted doublets in a 2D embedding.

    verbose : bool, optional (default: True)
        If True, print summary statistics.
    '''

    if threshold is None:
        from skimage.filters import threshold_minimum
        try: threshold = threshold_minimum(adata.uns['simulated'].obs['score.doublet'].to_numpy())
        except:
            warning('failed to automatically identify doublet score threshold.')
            warning('run `call_doublets` with user-specified threshold again.')
            return adata
    
    ld_obs = adata.obs['score.doublet'].to_numpy()
    ld_sim = adata.uns['simulated'].obs['score.doublet'].to_numpy()
    se_obs = adata.obs['score.doublet.se'].to_numpy()
    Z = (ld_obs - threshold) / se_obs

    adata.obs['is.doublet'] = ld_obs > threshold
    adata.uns['scrublet']['z_score'] = Z
    adata.uns['scrublet']['threshold'] = threshold
    
    detected_doublet_rate = (ld_obs > threshold).sum() / float(len(ld_obs))
    detectable_doublet_fraction = (ld_sim > threshold).sum() / float(len(ld_sim))
    overall_doublet_rate = detected_doublet_rate / detectable_doublet_fraction

    if verbose:
        info('detected doublet rate: {:.1f} %'.format(100 * detected_doublet_rate))
        info('estimated detectable doublet fraction: {:.1f} %'.format(100 * detectable_doublet_fraction))
        info('overall doublet rate: {:.1f} %'.format(100 * overall_doublet_rate))
        
    return adata