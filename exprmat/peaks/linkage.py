
# Peak-to-gene linkage
# --------------------
# 
# We estimated a linkage score for each peak-gene pair using linear regression models, 
# based on recent work described in the SHARE-seq method [Ma et al., 2020]. For 
# each gene, we computed the Pearson correlation coefficient r between the gene 
# expression and the accessibility of each peak within 500 kb of the gene TSS. 
# For each peak, we then computed a background set of expected correlation coefficients 
# given properties of the peak by randomly sampling 200 peaks located on a different 
# chromosome to the gene, matched for GC content, accessibility, and sequence length 
# (MatchRegionStats function in Signac). We then computed the Pearson correlation 
# between the expression of the gene and the set of background peaks. A z-score 
# was computed for each peak as z = (r − µ)/σ, where µ was the background mean 
# correlation coefficient and σ was the standard deviation of the background 
# correlation coefficients for the peak. We computed a p-value for each peak using 
# a one-sided z-test, and retained peak-gene links with a p-value < 0.05 and a 
# Pearson correlation coefficient > 0.05 or < -0.05. This was performed using the 
# LinkPeaks function in Signac.

import logging
from typing import Callable, Literal
from pathlib import Path
import numpy as np
import rustworkx as rx
import scipy.sparse as sp

from exprmat.ansi import error, warning, info, pprog
from exprmat.snapatac import (
    AnnData, AnnDataSet, link_region_to_gene, spearman
)


class node:
    def __init__(self, id: str = "", type: str = "") -> None:
        self.id = id
        self.type = type
        self.regr_fitness = None

    def __repr__(self):
        return str(self.__dict__)


class linkage:
    def __init__(
        self,
        distance: int = 0,
        label: str | None = None,
    ) -> None:
        self.distance = distance
        self.label = label
        self.regr_score = None
        self.cor_score = None
    
    def __repr__(self):
        return str(self.__dict__)
   

def init_network_from_annotation(
    regions: list[str],
    anno_file: Path,
    upstream: int = 250000,
    downstream: int = 250000,
    id_type: Literal["gene_name", "gene_id", "transcript_id"] = "gene_name",
    coding_gene_only: bool = True,
) -> rx.PyDiGraph:
    
    """
    Build CRE-gene network from gene annotations.

    Link CREs to genes if they are close to genes' promoter regions.

    Parameters
    ----------
    regions
        A list of peaks/regions, e.g., `["chr1:100-1000", "chr2:55-222"]`.

    anno_file
        The GFF file containing the transcript level annotations.

    upstream
        Upstream extension to the transcription start site.

    downstream
        Downstream extension to the transcription start site.

    id_type
        "gene_name", "gene_id" or "transcript_id".

    coding_gene_only
        Retain only coding genes in the network.

    Returns
    -------
    rx.PyDiGraph:
        A network where peaks/regions point towards genes if they are within genes'
        regulatory domains.
    """
    
    region_added = {}
    graph = rx.PyDiGraph()
    links = link_region_to_gene(
        regions,
        str(anno_file),
        upstream,
        downstream,
        id_type,
        coding_gene_only,
    )

    for (id, type), regions in links.items():
        to = graph.add_node(node(id.upper(), type))
        for i, t, distance in regions:
            key = (i, t)
            if key in region_added:
                graph.add_edge(region_added[key], to, linkage(distance))
            else: region_added[key] = graph.add_parent(to, node(i, t), linkage(distance))
    
    return graph


def add_cor_scores(
    network: rx.PyDiGraph,
    *,
    gene_mat: AnnData | AnnDataSet | None = None,
    peak_mat: AnnData | AnnDataSet | None = None,
    select: list[str] | None = None,
    overwrite: bool = False,
):
    """
    Compute correlation scores for any two connected nodes in the network.

    This function can be used to compute correlation scores for any type of
    associations. There are typically three types of edges in the network:

    1. Region -> gene: CREs regulate target genes.
    2. Gene -> gene: genes regulate other genes.
    3. Gene -> region: TFs bind to CREs.

    Parameters
    ----------
    select
        Run this for selected genes only.

    overwrite
        Whether to overwrite existing records.
    """

    key = "cor_score"
    if list(peak_mat.obs_names) != list(gene_mat.obs_names):
        error("gene matrix and peak matrix should have the same obs_names")
    if select is not None: select = set(select)
    without_overwrite = None if overwrite else key 

    if network.num_edges() > 0:
        data = get_data_iterator(network, peak_mat, gene_mat, select, without_overwrite)
        for (nd_X, X), (nd_y, y) in pprog(data, desc = 'processing for features'):
            if sp.issparse(X): X = X.todense()
            if sp.issparse(y): y = y.todense()
            scores = np.ravel(spearman(X.T, y.reshape((1, -1))))
            for nd, sc in zip(nd_X, scores):
                setattr(network.get_edge_data(nd, nd_y), key, sc)


def add_regr_scores(
    network: rx.PyDiGraph,
    *,
    peak_mat: AnnData | AnnDataSet | None = None,
    gene_mat: AnnData | AnnDataSet | None = None,
    select: list[str] | None = None,
    method: Literal["gb_tree", "elastic_net"] = "elastic_net",
    scale_X: bool = False,
    scale_Y: bool = False,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    use_gpu: bool = False,
    overwrite: bool = False,
):
    """
    Perform regression analysis for nodes and their parents in the network.

    Parameters
    ----------
    method
        Regresson model.

    scale_X
        Whether to scale the features.

    scale_Y
        Whether to scale the response variable.

    alpha
        Constant that multiplies the penalty terms in 'elastic_net'.

    l1_ratio
        Used in 'elastic_net'. The ElasticNet mixing parameter,
        with `0 <= l1_ratio <= 1`. For `l1_ratio = 0` the penalty is an L2 penalty.
        For `l1_ratio = 1` it is an L1 penalty. For `0 < l1_ratio < 1`,
        the penalty is a combination of L1 and L2.

    use_gpu
        Whether to use gpu
    """

    key = "regr_score"
    if peak_mat is not None and gene_mat is not None:
        if list(peak_mat.obs_names) != list(gene_mat.obs_names):
            error("gene matrix and peak matrix should have the same obs_names")
    if select is not None:
        select = set(select)
    without_overwrite = None if overwrite else key 
    tree_method = "gpu_hist" if use_gpu else "hist"
    
    if network.num_edges() == 0:
        return network

    for (nd_X, X), (nd_y, y) in pprog(get_data_iterator(
        network, peak_mat, gene_mat, select, without_overwrite, scale_X, scale_Y
    ), desc = 'processing for features ...'):
        y = np.ravel(y.todense()) if sp.issparse(y) else y
        if method == "gb_tree":
            scores, fitness = gradient_boost_tree(X, y, tree_method = tree_method)
        elif method == "elastic_net":
            scores, fitness = elastic_net(X, y, alpha, l1_ratio)
        elif method == "logistic_regression":
            scores, fitness = logistic_regr(X, y)
        else: error('unknown method specified. expected one of [gb_tree, elastic_net, logistic_regression]')
        network[nd_y].regr_fitness = fitness
        for nd, sc in zip(nd_X, scores):
            setattr(network.get_edge_data(nd, nd_y), key, sc)


def prune_network(
    network: rx.PyDiGraph,
    node_filter: Callable[[node], bool] | None = None,
    edge_filter: Callable[[int, int, linkage], bool] | None = None,
    remove_isolates: bool = True,
) -> rx.PyDiGraph:
    
    graph = rx.PyDiGraph()
    
    node_retained = [
        nid for nid in network.node_indices()
        if node_filter is None or node_filter(network[nid])
    ]              
    node_indices = graph.add_nodes_from([network[nid] for nid in node_retained])
    node_index_map = dict(zip(node_retained, node_indices))
   
    edge_retained = [
        (node_index_map[fr], node_index_map[to], data)
        for fr, to, data in network.edge_index_map().values()
        if fr in node_index_map and to in node_index_map and
            (edge_filter is None or edge_filter(fr, to, data))
    ]

    graph.add_edges_from(edge_retained)

    if remove_isolates:
        remove = []
        for nid in graph.node_indices():
            if graph.in_degree(nid) + graph.out_degree(nid) == 0:
                remove.append(nid)
        if len(remove) > 0:
            graph.remove_nodes_from(remove)
            info("removed {} isolated nodes.".format(len(remove)))

    return graph


class paired_iterator:

    def __init__(
        self,
        mat_X,
        mat_Y,
        idx_map_X,
        id_XY,
    ) -> None:
        
        self.mat_X = mat_X
        self.mat_Y = mat_Y
        self.idx_map_X = idx_map_X
        self.id_XY = id_XY
        self.index = 0

    def __len__(self):
        return self.mat_Y.shape[1]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration

        nd_X, nd_y = self.id_XY[self.index]
        y = self.mat_Y[:, self.index]
        X = self.mat_X[:, [self.idx_map_X[nd] for nd in nd_X]]

        self.index += 1
        return (nd_X, X), (nd_y, y)


def get_data_iterator(
    network: rx.PyDiGraph,
    peak_mat: AnnData | AnnDataSet | None,
    gene_mat: AnnData | AnnDataSet | None,
    select: set[str] | None = None,
    without_overwrite: str | None = None,
    scale_X: bool = False,
    scale_Y: bool = False,
) -> paired_iterator:
    
    from scipy.stats import zscore
    def get_mat(nids, node_getter, gene_mat, peak_mat):

        genes = []
        peaks = []
        mats = []

        for x in nids:
            nd = node_getter(x) 
            if nd.type == "gene" or nd.type == "motif": genes.append(x)
            elif nd.type == "region": peaks.append(x)
            else: raise NameError("unknown type: {}".format(nd.type))

        if len(genes) != 0:
            if len(genes) == gene_mat.n_vars:
                mats.append(gene_mat.X[:])
            else:
                idx_map = {x.upper(): i for i, x in enumerate(gene_mat.var_names)}
                ix = [idx_map[node_getter(x).id] for x in genes]
                mats.append(gene_mat.X[:, ix])

        if len(peaks) != 0:
            if len(peaks) == peak_mat.n_vars:
                mats.append(peak_mat.X[:])
            else:
                if peak_mat.isbacked: ix = peak_mat.var_ix([node_getter(x).id for x in peaks])
                else: ix = [peak_mat.var_names.get_loc(node_getter(x).id) for x in peaks]
                mats.append(peak_mat.X[:, ix])
        
        if all([sp.issparse(x) for x in mats]): mat = sp.hstack(mats, format = "csc")
        else: mat = np.hstack(mats)
        return (genes + peaks, mat)

    all_genes = set([x.upper() for x in gene_mat.var_names])
    select = all_genes if select is None else select
    id_XY = []

    for nid in network.node_indices():
        if network[nid].type == "region" or network[nid].id in select:
            parents = [
                pid for pid, _, edge_data in network.in_edges(nid)
                if (without_overwrite is None or
                    getattr(edge_data, without_overwrite) is None) and
                    (network[pid].type == "region" or network[pid].id in all_genes)
            ]

            if len(parents) > 0: id_XY.append((parents, nid))
    
    unique_X = list({y for x, _ in id_XY for y in x})
    id_XY, mat_Y = get_mat(id_XY, lambda x: network[x[1]], gene_mat, peak_mat)
    unique_X, mat_X = get_mat(unique_X, lambda x: network[x], gene_mat, peak_mat)

    if scale_X: mat_X = zscore(mat_X, axis = 0)
    if scale_Y: mat_Y = zscore(mat_Y, axis=0)
        
    return paired_iterator(
        mat_X, mat_Y, {v: i for i, v in enumerate(unique_X)}, id_XY
    )


def logistic_regr(X, y):
    from sklearn.linear_model import LogisticRegression 
    y = y != 0
    regr = LogisticRegression(max_iter = 1000, random_state = 0).fit(X, y)
    return np.ravel(regr.coef_), regr.score(X, y)


def elastic_net(X, y, alpha = 1, l1_ratio = 0.5, positive = False):
    from sklearn.linear_model import ElasticNet
    X = np.asarray(X)
    y = np.asarray(y)
    regr = ElasticNet(
        alpha = alpha, l1_ratio = l1_ratio, positive = positive,
        random_state = 0, copy_X = False, max_iter = 10000,
    ).fit(X, y)
    return regr.coef_, regr.score(X, y)


def gradient_boost_tree(X, y, tree_method = "hist"):
    import xgboost as xgb
    regr = xgb.XGBRegressor(tree_method = tree_method).fit(X, y)
    return regr.feature_importances_, regr.score(X, y)


def graph_to_linkage(graph, assembly):
    
    peak_chr = []
    peak_start = []
    peak_end = []
    intensities = []
    regression = []
    gene = []

    for i, (a, b, link) in graph.edge_index_map().items():
        node1 = graph.get_node_data(a)
        node2 = graph.get_node_data(b)

        if node1.type != node2.type:
            intensities.append(link.cor_score)
            regression.append(link.regr_score)
            if node1.type == 'region':
                peak_chr.append(node1.id.split(':')[0])
                peak_start.append(int(node1.id.split(':')[1].split('-')[0]))
                peak_end.append(int(node1.id.split(':')[1].split('-')[1]))
                gene.append(node2.id)
            else: 
                peak_chr.append(node2.id.split(':')[0])
                peak_start.append(int(node2.id.split(':')[1].split('-')[0]))
                peak_end.append(int(node2.id.split(':')[1].split('-')[1]))
                gene.append(node1.id)
    
    import pandas as pd
    data = pd.DataFrame({
        'chr': peak_chr,
        'start': peak_start,
        'end': peak_end,
        'id': gene,
        'cor': intensities,
        'regr': regression
    })

    # querying assembly for genomic locations
    from exprmat.data.finders import get_genome_model
    gmodel = get_genome_model(assembly)
    gmodel = gmodel.loc[
        gmodel['type'] == 'gene', 
        ['id', 'start', 'end', 'strand']
    ].set_index('id').copy()
    
    gmodel['tss'] = [
        start if strand == '+' else end
        for start, end, strand in zip(gmodel['start'], gmodel['end'], gmodel['strand'])
    ]

    del gmodel['start']
    del gmodel['end']
    del gmodel['strand']
    data = data.join(gmodel, on = 'id', how = 'left')
    return data


def gene_peak_linkage(
    rna, peaks, rna_lognorm = 'X', peak_norm = 'X',
    upstream = 250000, downstream = 250000,
    method = 'elastic_net', scale_X = True, scale_Y = True, alpha = 1, l1_ratio = 0.5,
    prune_correlation = 0.05,
):

    from exprmat.data.finders import get_genome_gff_fname, get_genome_model

    # filtering for shared cells
    qcells = rna.obs_names.tolist()
    atac_mask = [x in qcells for x in peaks.obs_names]
    indices = peaks.obs_names[atac_mask]
    peak_matrix = peaks[indices, :].copy()
    expr_matrix = rna[indices, :].copy()

    # after filtering for cells, we should remove genes or peaks with zero expression.
    import exprmat.reader.static as st
    peak_matrix = st.adata_filter_column_by_sum(peak_matrix, None, layer = peak_norm)
    expr_matrix = st.adata_filter_column_by_sum(expr_matrix, None, layer = rna_lognorm)

    # rename features
    from exprmat.utils import choose_layer
    peak_matrix.X = choose_layer(peak_matrix, layer = peak_norm)
    expr_matrix.X = choose_layer(expr_matrix, layer = rna_lognorm)
    peak_matrix.var_names = peak_matrix.var_names.str.replace(f'peak:{peaks.uns["assembly"]}:', '')
    del peak_matrix.var
    expr_matrix = expr_matrix[:, ~expr_matrix.var['id'].isna()].copy()
    expr_matrix.var_names = expr_matrix.var['id'].tolist()
    del expr_matrix.var
    del peak_matrix.layers
    del expr_matrix.layers

    # must be dense arrays
    import scipy.sparse as sp
    import numpy as np
    if sp.issparse(peak_matrix.X): peak_matrix.X = np.array(peak_matrix.X.astype('float32').todense())
    if sp.issparse(expr_matrix.X): expr_matrix.X = np.array(expr_matrix.X.astype('float32').todense())

    graph = init_network_from_annotation(
        [x.replace(f'peak:{peaks.uns["assembly"]}:', '') for x in peak_matrix.var_names],
        get_genome_gff_fname(peaks.uns['assembly']),
        id_type = 'gene_id', upstream = upstream, downstream = downstream
    )

    info('calculating correlations ...')
    add_cor_scores(
        graph, 
        gene_mat = expr_matrix,
        peak_mat = peak_matrix
    )

    import warnings
    warnings.filterwarnings('ignore')
    info('calculating regression ...')
    add_regr_scores(
        graph, 
        gene_mat = expr_matrix,
        peak_mat = peak_matrix,
        method = method,
        scale_X = scale_X,
        scale_Y = scale_Y,
        alpha = alpha,
        l1_ratio = l1_ratio,
        use_gpu = False,
        overwrite = True
    )

    warnings.filterwarnings('default')
    info(f'pruning network for correlation >= {prune_correlation} ...')
    pruned = prune_network(
        graph,
        node_filter = lambda n: True,
        edge_filter = lambda a, b, l: (l.cor_score >= prune_correlation) if l.cor_score else False,
        remove_isolates = True
    )

    return graph_to_linkage(pruned, peaks.uns['assembly'])