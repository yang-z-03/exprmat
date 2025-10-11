
import numpy as np
import pandas as pd
import anndata as ad
from exprmat import error, warning, pprog
import exprmat.reader.static as st


def construct_graph(adata, trajectory_key = 'ppt'):

    import igraph

    graph = igraph.Graph(
        edges = [[a, b] for a, b in zip(
            adata.uns[f'{trajectory_key}.graph']['pp_seg']['from'].tolist(),
            adata.uns[f'{trajectory_key}.graph']['pp_seg']['to'].tolist()
        )]
    )

    return graph


def shortest_paths(adata, destination, source = None, trajectory_key = 'ppt'):

    from exprmat import warning, error
    if source is None:
        source = adata.uns[f'{trajectory_key}.graph']['root']
    
    graph = construct_graph(adata, trajectory_key)
    paths = graph.get_shortest_paths(source, to = destination, output = "vpath")
    if len(paths) > 0: warning('there exists several shortest paths. returing a random one.')
    path = paths[0]

    segments = adata.uns[f'{trajectory_key}.graph']['pp_seg'].copy()
    segment_path = []
    startings = []
    endings = []
    orientation = []
    for x in range(len(path) - 1):
        start = path[x]
        end = path[x + 1]
        startings.append(start)
        endings.append(end)
        
        search1 = (segments['from'] == start) & (segments['to'] == end)
        search2 = (segments['from'] == end) & (segments['to'] == start)
        if search1.sum() + search2.sum() <= 0:
            error('could not find path identity. corrupted data suspected')
        elif search1.sum() == 1:
            segment_path.append(segments.loc[search1, 'n'].iloc[0])
            orientation.append(True)
        elif search2.sum() == 1:
            segment_path.append(segments.loc[search2, 'n'].iloc[0])
            orientation.append(False)
        else: error(f'duplicated segments in metadata (from {start} to {end})')
    
    return {
        'segments': segment_path,
        'starts': startings,
        'ends': endings,
        'orient': orientation
    }


def sort_pseudotime(adata, segment, data = 'X', reversed = False, trajectory_key = 'ppt'):
    
    from exprmat.utils import choose_layer
    import anndata as ad

    matrix = choose_layer(adata, layer = data)
    trimmed = ad.AnnData(X = matrix, obs = adata.obs, var = adata.var)
    trimmed = trimmed[trimmed.obs[f'{trajectory_key}.seg'] == segment, :].copy()
    order = trimmed.obs[f'{trajectory_key}.pseudotime'].sort_values(ascending = not reversed).index.tolist()
    order = trimmed[order, :].copy()
    return order


def binned_statistics(sorted_pstime, trajectory_key = 'ppt', statistic = 'mean', bins = 100, reversed = False):

    from scipy.stats import binned_statistic
    from scipy.sparse import issparse
    import numpy as np

    ptimes = sorted_pstime.obs[f'{trajectory_key}.pseudotime'].to_numpy()
    uniques = sorted_pstime.obs[f'{trajectory_key}.pseudotime'].unique()

    X = sorted_pstime.X.T
    if issparse(X): X = X.todense()
    X = np.array(X)
    minx = sorted_pstime.obs[f'{trajectory_key}.pseudotime'].min()
    maxx = sorted_pstime.obs[f'{trajectory_key}.pseudotime'].max()

    binedges = []
    statistics = []
    for sep in range(bins):

        start = minx + (maxx - minx) * sep / bins
        end = minx + (maxx - minx) * (sep + 1) / bins
        binedges.append(start)
        have_one = (uniques >= start) & (uniques < end)

        if have_one.sum() == 0:
            statistics.append(statistics[-1] if len(statistics) > 0 else np.zeros(X.shape[0]))

        else:
            mask = (ptimes >= start) & (ptimes < end)
            masked = X[:, mask].mean(1).T
            statistics.append(masked)
    
    binedges.append(maxx)

    binedges = np.array(binedges)
    statistics = np.array(statistics).T

    if reversed:
        return {
            'statistic': statistics[:, ::-1],
            'edges': binedges[::-1]
        }
    
    return {
        'statistic': statistics,
        'edges': binedges
    }


def regress(binned, order = 3, desc = None):

    from sklearn.preprocessing import PolynomialFeatures as polynomial
    from sklearn.linear_model import LinearRegression as linregress
    import statsmodels.api as sm
    import numpy as np
    from exprmat import pprog

    significance = []
    preds = []
    model_pvals = []
    model_params = []
    
    x = np.array([binned['edges'][:-1]]).T
    poly_features = polynomial(degree = order, include_bias = True)
    poly = poly_features.fit_transform(x)

    for n in pprog(range(binned['statistic'].shape[0]), desc = desc if desc else 'testing for genes'):

        poly_features = polynomial(degree = order, include_bias = True)
        y = binned['statistic'][n : n + 1, :].T
        model = linregress()
        model.fit(poly, y)

        pred = model.predict(poly)
        X = sm.add_constant(poly)
        model = sm.OLS(y, X).fit()
        significance.append(model.f_pvalue)
        preds.append(pred.T[0])
        model_pvals.append(model.pvalues)
        model_params.append(model.params)
    
    return (
        np.array(significance),
        np.array(preds),
        np.array(model_pvals),
        np.array(model_params)
    )


def cluster_features(adata, a2d, areal, pseudotime, pvals):

    a2d[a2d < 0] = 0
    scaled_data = a2d / (a2d.max(1, keepdims = True) + 1e-5)

    # use another clustering method. let's use leiden

    scaled_data[np.isnan(scaled_data)] = 0
    feats = scaled_data.copy()

    dummy_cluster = ad.AnnData(X = feats, obsm = { 'features': feats })
    dummy_proceedings = ad.AnnData(X = feats.T, obsm = { 'features': feats.T })

    st.rna_knn(
        dummy_cluster, None,  
        use_rep = 'features',
        n_comps = None,
        n_neighbors = 30,
    )

    st.rna_knn(
        dummy_proceedings, None,  
        use_rep = 'features',
        n_comps = None,
        n_neighbors = 30,
    )

    st.rna_leiden(
        dummy_cluster, None,
        resolution = 3,
        restrict_to = None,
        random_state = 42,
        key_added = 'leiden',
        adjacency = None,
        directed = None,
        use_weights = True,

        # for gpu implementation, the n_iterations have different meanings than the igraph's
        # cpu implementation. if you set to n_iter = 2, this will give a far-from-converged
        # result with hundreds of thousands of clusters.

        n_iterations = 200,
        partition_type = None,
        neighbors_key = None,
        obsp = None,
        flavor = 'igraph',
        use_gpu = True
    )

    st.rna_leiden(
        dummy_proceedings, None,
        resolution = 3,
        restrict_to = None,
        random_state = 42,
        key_added = 'leiden',
        adjacency = None,
        directed = None,
        use_weights = True,

        # for gpu implementation, the n_iterations have different meanings than the igraph's
        # cpu implementation. if you set to n_iter = 2, this will give a far-from-converged
        # result with hundreds of thousands of clusters.

        n_iterations = 200,
        partition_type = None,
        neighbors_key = None,
        obsp = None,
        flavor = 'igraph',
        use_gpu = True
    )

    adata = ad.AnnData(X = a2d, obs = adata.var, layers = {'scaled': scaled_data, 'expression': areal})
    adata.obs['leiden'] = dummy_cluster.obs['leiden'].tolist()
    adata.var['leiden'] = dummy_proceedings.obs['leiden'].tolist()
    adata.var['pseudotime'] = pseudotime
    adata.obs['p'] = pvals

    # sort using dendrogram

    # from scipy.cluster.hierarchy import dendrogram, linkage

    # ordered = adata[
    #     adata.obs['leiden'].sort_values().index, :
    # ].copy()

    # # merge average leiden intensity
    # # genes (rows) first.

    # avgs = []
    # for l in range(len(ordered.obs['leiden'].unique())):
    #     avgs.append(np.array(ordered[ordered.obs['leiden'] == str(l), :].X.mean(0)))

    # avgs = np.stack(avgs)
    # lnk = linkage(avgs, method = 'ward')
    # leaves = dendrogram(
    #     lnk,
    #     truncate_mode = 'level',
    #     leaf_rotation = 90,
    #     leaf_font_size = 8,
    #     get_leaves = True,
    #     no_plot = True
    # )

    # leiden = ordered.obs['leiden'].tolist()
    # indices = [x for x in range(len(leiden))]
    # sort_table = {}

    # for l, i in zip(leiden, indices):
    #     if not l in sort_table.keys(): sort_table[l] = []
    #     sort_table[l] += [i]

    # gene_index = []
    # for k in leaves['ivl']: gene_index += sort_table[k]
    # ordered = ordered[gene_index, :].copy()

    # sort by leiden's gravity center

    ordered = adata[
        adata.obs['leiden'].sort_values().index, :
    ].copy()

    leiden = ordered.obs['leiden'].tolist()
    indices = [x for x in range(len(leiden))]
    gravities = {}
    sort_table = {}

    for l, i in zip(leiden, indices):
        if not l in gravities.keys(): gravities[l] = []
        if not l in sort_table.keys(): sort_table[l] = []
        sort_table[l] += [i]

    for l in range(len(ordered.obs['leiden'].unique())):
        gravities[str(l)] = np.argmax(np.array(
            ordered[ordered.obs['leiden'] == str(l), :].X.mean(0)
        ))

    sorted_dict = dict(sorted(gravities.items(), key = lambda item: item[1]))
    gene_index = []
    for k in sorted_dict.keys(): gene_index += sort_table[k]
    ordered = ordered[gene_index, :].copy()

    return ordered


def trace_trajectory(
    adata, destination, source = None, 
    trajectory_key = 'ppt', lognorm = 'X', 
    binning = 100, statistic = 'mean',
    skip_sparsity = 40,
    order = 4, key_added = 'trace'
):

    path = shortest_paths(
        adata, destination = destination, source = source, 
        trajectory_key = trajectory_key
    )

    # calculate for each segments
    segments = path['segments']
    orientation = path['orient']

    xs = []
    preds = []
    expression = []
    pvals = []

    for seg, ori in zip(segments, orientation):

        data = sort_pseudotime(
            adata, segment = seg, data = lognorm, 
            reversed = not ori, trajectory_key = trajectory_key
        )

        n_uniques = len(data.obs[f'{trajectory_key}.pseudotime'].unique())
        if skip_sparsity and (n_uniques <= skip_sparsity):
            continue
        
        binned = binned_statistics(
            data, trajectory_key = trajectory_key, 
            bins = binning, statistic = statistic, reversed = not ori
        )

        x_sig, x_preds, _, _ = regress(
            binned, order = order, 
            desc = f'regressing genes: {seg}'
        )

        xs.append(binned['edges'][:-1]) 
        preds.append(x_preds)
        expression.append(binned['statistic'])
        pvals.append(x_sig)
    
    xs = np.concat(xs, axis = 0)
    preds = np.concat(preds, axis = 1)
    expression = np.concat(expression, axis = 1)
    pvals = np.stack(pvals, axis = 1)
    pvals = np.nanmin(pvals, axis = 1)
    ordered = cluster_features(adata, preds, expression, xs, pvals)

    adata.uns[key_added] = ordered
    return ordered