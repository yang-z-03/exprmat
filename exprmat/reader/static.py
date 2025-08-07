
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from collections import defaultdict
import os

from exprmat.data.finders import get_genome
from exprmat.reader.matcher import attach_tcr
from exprmat.ansi import warning, info, error, red, green
from exprmat.configuration import default as cfg


def search_genes(genes):
    species_db = {}
    columns = {}
    n_genes = 0

    for gene in genes:
        if not gene.startswith('rna:'):
            error('gene name within rna modality does not start with prefix `rna:`')
        modal, taxa, ugene = gene.split(':')
        if not taxa in species_db.keys(): species_db[taxa] = get_genome(taxa)

        gene_meta = species_db[taxa].loc[ugene]
        for k in gene_meta.index.tolist():
            if k in columns.keys(): columns[k].append(gene_meta[k])
            elif n_genes == 0: columns[k] = [gene_meta[k]]
            else: columns[k] = [None] * n_genes + [gene_meta[k]]
        
        n_genes += 1
    
    variables = pd.DataFrame(columns)
    variables.index = genes
    return variables
    

def remove_slot(
    adata, sample_name, slot, names
):
    if slot not in ['obs', 'var', 'obsm', 'varm', 'obsp', 'varp', 'layers', 'uns']:
        error(f'unsupported slot in annotated data: `{slot}`')
    
    for name in names:
        ref = None
        if slot == 'obs': ref = adata.obs
        if slot == 'var': ref = adata.var
        if slot == 'obsm': ref = adata.obsm
        if slot == 'varm': ref = adata.varm
        if slot == 'obsp': ref = adata.obsp
        if slot == 'varp': ref = adata.varp
        if slot == 'layers': ref = adata.layers
        if slot == 'uns': ref = adata.uns

        if ref is None:
            error(f'unsupported slot in annotated data: `{slot}`')
        
        if name not in ref.keys():
            warning(f'`{name}` does not exist in slot `{slot}`, skipped operation')
        else: 
            info(f'deleted `{name}` from slot `{slot}`')
            del ref[name]


def rna_qc(
    adata, sample_name, mt_seqid = 'MT',
    mt_percent = 0.15,
    ribo_genes = None,
    ribo_percent = None,
    outlier_mode = 'mads',
    outlier_n = 5,
    doublet_method = 'scrublet',
    min_cells = 3,
    min_genes = 300
):
    from exprmat.preprocessing.qc import rna_qc as _rna_qc
    _rna_qc(
        adata, sample = sample_name, mt_seqid = mt_seqid,
        mt_percent = mt_percent,
        ribo_genes = ribo_genes,
        ribo_percent = ribo_percent,
        outlier_mode = outlier_mode,
        outlier_n = outlier_n,
        doublet_method = doublet_method,
        min_cells = min_cells,
        min_genes = min_genes
    )


def rna_filter(adata, sample_name):
    qc_cells = adata[adata.obs['qc'], adata.var['qc']].copy()
    # raw is rubbish. it does seldom over just throwing it.
    # qc_cells.raw = adata
    return qc_cells


def rna_log_normalize(
    adata, sample_name, key_source = 'X', 
    key_norm = 'norm', key_lognorm = 'lognorm', **kwargs
):
    from exprmat.preprocessing import log_transform, normalize
    normalize(adata, counts = key_source, dest = key_norm, method = 'total', **kwargs)
    log_transform(adata, norm = key_norm, dest = key_lognorm)
    if key_source == 'X': adata.layers['counts'] = adata.X
    adata.X = adata.layers[key_lognorm]


def rna_select_hvg(adata, sample_name, key_lognorm = 'lognorm', method = 'vst', **kwargs):
    from exprmat.preprocessing import highly_variable
    highly_variable(
        adata, 
        counts = 'counts', lognorm = key_lognorm, 
        method = method, **kwargs
    )


def rna_scale_pca(
    adata, sample_name, key_added = 'pca', n_comps = 50, 
    hvg = 'vst.hvg', key_lognorm = 'lognorm', key_scaled = 'scaled', **kwargs):
    if hvg not in adata.var.keys():
        warning('you should select highly variable genes before pca reduction.')
        warning('if you really want to run an all genes, you should manually confirm your choice by')
        warning(f'adding a var slot `hvg` (by default `{hvg}`) with all true values manually.')
        error('now, we stop your routine unless you know what you are doing.')
    
    hvg_subset = adata[:, adata.var[hvg]].copy()
    from exprmat.preprocessing import scale
    from exprmat.reduction import run_pca
    from exprmat.utils import align
    scale(hvg_subset, lognorm = key_lognorm, dest = key_scaled)
    run_pca(hvg_subset, key_added = key_added, layer = key_scaled, n_comps = n_comps, **kwargs)
    
    adata.obsm[key_added] = hvg_subset.obsm[key_added]
    hvg_names = hvg_subset.var_names.tolist()
    all_names = adata.var_names.tolist()
    indices = np.array(align(hvg_names, all_names))

    adata.uns[key_added] = {
        'variance': hvg_subset.uns[key_added]['variance'],
        'pct.variance': hvg_subset.uns[key_added]['pct.variance'],
        'singular': hvg_subset.uns[key_added]['singular'],
        'params': hvg_subset.uns[key_added]['params']
    }
    
    # set the principle components back to the parent.
    pc = np.zeros((adata.n_vars, n_comps))
    pc[indices, :] = hvg_subset.varm[key_added]
    adata.varm[key_added] = pc


def rna_knn(adata, sample_name, **kwargs):
    from exprmat.reduction import run_knn
    run_knn(adata, **kwargs)


def rna_leiden(adata, sample_name, **kwargs):
    from exprmat.clustering import run_leiden
    run_leiden(adata, **kwargs)


def rna_leiden_subcluster(
    adata, sample_name, cluster_key, clusters, 
    restrict_to = None, key_added = 'leiden', **kwargs
):
    from exprmat.clustering import run_leiden
    run_leiden(adata, restrict_to = (cluster_key, clusters), key_added = '.leiden.temp', **kwargs)
    temp = adata.obs['.leiden.temp'].tolist()
    orig = adata.obs[cluster_key].tolist()
    merge = [x if x not in clusters else y.replace(',', '.') for x, y in zip(orig, temp)]
    del adata.obs['.leiden.temp']
    del adata.uns['.leiden.temp']
    adata.obs[key_added] = merge

    # for all categorical types:
    adata.obs[key_added] = \
        adata.obs[key_added].astype('category')
    print(adata.obs[key_added].value_counts())


def rna_umap(adata, sample_name, **kwargs):
    from exprmat.reduction import run_umap
    run_umap(adata, **kwargs)


def rna_mde(adata, sample_name, data = 'pca', key_added = 'mde', **kwargs):
    from exprmat.reduction.mde import mde
    emb = mde(adata.obsm[data], **kwargs)
    adata.obsm[key_added] = emb


def rna_mde_fit(
    adata, sample_name, data = 'pca', 
    based = 'umap', mask_key = 'sample', mask_values = [], 
    key_added = 'mde', **kwargs
):
    from exprmat.reduction.mde import mde_fit
    mask = [x in mask_values for x in adata.obs[mask_key]]
    emb = mde_fit(adata.obsm[data], fit = adata.obsm[based], fit_mask = mask, **kwargs)
    adata.obsm[key_added] = emb


def rna_markers(adata, sample_name, **kwargs):
    from exprmat.descriptive.de import markers
    markers(adata, **kwargs)


def rna_markers_deseq(
    adata, sample_name, counts = 'counts', 
    metadata = ['sample', 'group'], formula = '~ group',
    variable = 'group', experiment = '.', control = '.', 
    statistics_params = {'quiet': True},
    key_added = 'markers',
    **kwargs
):
    
    from exprmat.deseq.dataset import deseq_dataset
    from exprmat.deseq.stats import deseq_stats
    from exprmat.utils import choose_layer

    maybe_sparse = choose_layer(adata, layer = counts)
    from scipy.sparse import issparse
    if issparse(maybe_sparse):
        maybe_sparse = maybe_sparse.todense()
    
    if isinstance(maybe_sparse, np.matrix):
        maybe_sparse = np.array(maybe_sparse)

    # deseq can only apply on integral values.
    maybe_sparse = np.floor(maybe_sparse)

    # build deseq dataset
    annd = ad.AnnData(
        X = maybe_sparse,
        obs = adata.obs[metadata]
    )

    dds = deseq_dataset(adata = annd, design = formula, **kwargs)
    dds.deseq2()

    dstat = deseq_stats(dds, [variable, experiment, control], **statistics_params)
    dstat.summary()

    stat = dstat.results_df.copy()
    stat.columns = ['mean', 'lfc', 'lfc.se', 'scores', 'p', 'q']
    stat['names'] = adata.var_names.tolist()
    stat['gene'] = adata.var['gene'].tolist()
    stat = stat.loc[~np.isnan(stat['mean']), :].copy()
    stat.loc[np.isnan(stat['q']), 'q'] = 1

    with np.errstate(divide = 'ignore'):
        stat['log10.p'] = -np.log10(stat['p'])
        stat['log10.q'] = -np.log10(stat['q'])

    # attach differential gene to uns
    adata.uns[key_added] = {
        'params': {'reference': control},
        'differential': {
            experiment: stat
        }
    }


def rna_kde(adata, sample_name, **kwargs):
    from exprmat.descriptive.kde import density
    density(adata, **kwargs)


def rna_proportion(
    adata, sample_name, major, minor, normalize = 'columns'
):
    if normalize == 'major': normalize = 'columns'
    if normalize == 'minor': normalize = 'index'
    tab = pd.crosstab(adata.obs[major], adata.obs[minor], normalize = normalize)
    return tab


def rna_infercnv(
    adata, sample_name, inplace = None, **kwargs
):
    from exprmat.cnv.infercnv import infercnv
    chr_position, cnv_matrix, pergene = infercnv(adata, inplace = False, **kwargs)
    key = kwargs.get('key_added', 'cnv')
    adata.obsm[key] = cnv_matrix
    adata.uns[key] = { 'chr.pos': chr_position }
    if pergene is not None:
        adata.layers[key] = pergene


def rna_summary(
    adata, sample_name, data = 'X', method = 'n', method_args = {},
    orient = 'obs', on = 'sample', across = None, split = None, 
    attached_metadata_on = None, attached_metadata_across = None,
    attach_method_on = 'first', attach_method_across = 'first'
):
    from exprmat.descriptive.summary import summarize
    return summarize(
        adata, data = data, method = method, method_args = method_args,
        orient = orient, on = on, across = across, split = split,
        attached_metadata_on = attached_metadata_on,
        attached_metadata_across = attached_metadata_across,
        attach_method_on = attach_method_on,
        attach_method_across = attach_method_across
    )


def rna_aggregate(
    adata, sample_name, data = 'X', method = 'mean', method_args = {},
    obs_key = 'sample', var_key = None
):
    from exprmat.descriptive.aggregate import aggregate
    return aggregate(
        adata, data = data, method = method, method_args = method_args,
        obs_key = obs_key, var_key = var_key
    )


def rna_attach_tcr(adata, sample_name, searchdir):
    # automatically search the tcr folder in the root directory.
    for fpath in os.listdir(searchdir):
        if not (fpath.endswith('.tsv') or fpath.endswith('.tsv.gz')): continue
        attach_tcr(adata, os.path.join(searchdir, fpath))
    
    assert 'clone.id' in adata.obs.columns
    assert 'clone' in adata.obs.columns
    assert 'tra' in adata.obs.columns
    assert 'trb' in adata.obs.columns
    n_match = adata.n_obs - adata.obs['clone'].isna().sum()
    info(f'{n_match} out of {adata.n_obs} ({(100 * n_match / adata.n_obs):.1f}%) tcr detections mapped.')


def rna_calculate_tcr_metrics(adata, sample_name, expanded_clone = 2, size_stat = 'clone.id'):
    assert 'clone.id' in adata.obs.columns
    assert 'clone' in adata.obs.columns
    assert 'tra' in adata.obs.columns
    assert 'trb' in adata.obs.columns

    # valid tcr a and b:
    empty_tcra = [(x == 'na') or ('nt(na)' in x) for x in adata.obs['tra'].tolist()]
    empty_tcrb = [(x == 'na') or ('nt(na)' in x) for x in adata.obs['trb'].tolist()]
    adata.obs['trab'] = [not(x or y) for x, y in zip(empty_tcra, empty_tcrb)]

    # expanded tcr clone
    if 'tcr.expanded' in adata.obs.columns: del adata.obs['tcr.expanded']
    if 'tcr.clone.size' in adata.obs.columns: del adata.obs['tcr.clone.size']
    if 'tcr.clone.sum' in adata.obs.columns: del adata.obs['tcr.clone.sum']
    if 'tcr.clone.size.rel' in adata.obs.columns: del adata.obs['tcr.clone.size.rel']

    adata.obs['tcr.clone.sum'] = 0
    for samp in adata.obs['sample'].unique():
        adata.obs.loc[adata.obs['sample'] == samp, 'tcr.clone.sum'] = \
            adata.obs.loc[adata.obs['sample'] == samp, 'trab'].sum()
    
    sizes = adata.obs[size_stat].value_counts()
    sizes = pd.DataFrame({
        'key': sizes.keys().tolist(),
        'tcr.clone.size': sizes.values
    })
    sizes.index = sizes['key'].tolist()
    del sizes['key']

    if 'na' in sizes.index:
        sizes.loc['na', 'tcr.clone.size'] = 0
    
    ljoin = adata.obs.join(sizes, on = size_stat, how = 'left')
    assert len(ljoin) == adata.n_obs
    adata.obs = ljoin

    adata.obs['tcr.expanded'] = adata.obs['tcr.clone.size'] > expanded_clone
    adata.obs['tcr.clone.size.rel'] = adata.obs['tcr.clone.size'] / adata.obs['tcr.clone.sum']


def rna_aggregate_tcr_by_identity(adata, sample_name, identity = 'patient'):
    
    cloneid = {}
    assert 'clone' in adata.obs.columns
    rawclone = adata.obs['clone'].tolist()
    patient = adata.obs[identity].tolist()
    clid = []

    for p, cl in zip(patient, rawclone):
        if p not in cloneid: cloneid[p] = {}
        if cl not in cloneid[p].keys(): cloneid[p][cl] = 'c:' + str(len(cloneid[p]) + 1)
        clid.append(p + ':' + cloneid[p][cl])
    
    adata.obs['clone.id.' + identity] = clid
    return


def rna_calculate_startracs_metrics(
    adata, sample_name, 
    clonotype = 'clone.id', cluster = 'leiden', tissue = None
):
    
    from exprmat.descriptive.tcr import (
        expansion, plasticity, transition, migration
    )

    expansion(adata, clonotype = clonotype, cluster = cluster)
    plasticity(adata, clonotype = clonotype, cluster = cluster)
    transition(adata, clonotype = clonotype, cluster = cluster)

    if tissue is not None:
        migration(adata, clonotype = clonotype, cluster = tissue)


def rna_calculate_startracs_pairwise_metrics(
    adata, sample_name, base,
    clonotype = 'clone.id', cluster = 'leiden', key_added = 'tcr.cluster.ptrans'
):
    
    from exprmat.descriptive.tcr import (
        pairwise_transition
    )

    pairwise_transition(adata, base, clonotype = clonotype, cluster = cluster, key = key_added)


def rna_expression_mask(adata, sample_name, gene, key, lognorm = 'X', threshold = 0.1, negate = False):
    from exprmat.utils import find_variable
    if not negate: adata.obs[key] = find_variable(adata, gene, layer = lognorm) >= threshold
    else: adata.obs[key] = find_variable(adata, gene, layer = lognorm) < threshold


def rna_gsea(
    adata, sample_name, taxa,
    # differential expression slots:
    de_slot, group_name = None,
    min_pct = 0.0, max_pct_reference = 1, 
    min_lfc = None, max_lfc = None, remove_zero_pval = False,
    max_q = 0.05,

    key_added = 'gsea',
    gene_sets = 'all',
    identifier = 'entrez'
):
    from exprmat.descriptive.gse import gse
    return gse(
        adata, taxa = taxa, de_slot = de_slot, group_name = None,
        min_pct = min_pct, max_pct_reference = max_pct_reference,
        min_lfc = min_lfc, max_lfc = max_lfc, remove_zero_pval = remove_zero_pval,
        key_added = key_added, gene_sets = gene_sets, identifier = identifier,
        max_q = max_q, 
    )


def rna_opa(
    adata, sample_name, taxa,
    # differential expression slots:
    de_slot, group_name = None,
    min_pct = 0.0, max_pct_reference = 1, 
    min_lfc = None, max_lfc = None, remove_zero_pval = False,
    use_abs_lfc = False, min_abs_lfc = 1.0, max_abs_lfc = 25.0,
    max_q = 0.05,

    key_added = 'gsea',
    gene_sets = 'all',
    identifier = 'entrez',
    opa_cutoff = 0.05,
    **kwargs
):
    from exprmat.descriptive.gse import opa
    return opa(
        adata, taxa = taxa, de_slot = de_slot, group_name = None,
        min_pct = min_pct, max_pct_reference = max_pct_reference,
        min_lfc = min_lfc, max_lfc = max_lfc, remove_zero_pval = remove_zero_pval,
        use_abs_lfc = use_abs_lfc, min_abs_lfc = min_abs_lfc, max_abs_lfc = max_abs_lfc,
        key_added = key_added, gene_sets = gene_sets, identifier = identifier,
        opa_cutoff = opa_cutoff, max_q = max_q, **kwargs
    )


def rna_gsva(
    adata, sample_name, taxa,
    identifier = 'uppercase', gene_sets = 'kegg', lognorm = 'X',
    n_cores = 1, kcdf = 'Gaussian', weight = 1, min_genes = 15, max_genes = 1000
):
    from exprmat.descriptive.gse import gsva
    gsva_df = gsva(
        adata, taxa = taxa, identifier = identifier, gene_sets = gene_sets,
        lognorm = lognorm, n_cores = n_cores, kcdf = kcdf,
        weight = weight, min_genes = min_genes, max_genes = max_genes
    )

    return gsva_df


def rna_ligand_receptor(
    adata, sample_name, flavor = 'ra', 
    taxa_source = 'hsa', taxa_dest = 'hsa',
    gene_symbol = None,
    groupby = 'cell.type', use_raw = False,
    min_cells = 5, expr_prop = 0.1,
    # set to a smaller value. lianapy uses 1000 by default.
    n_perms = 500, seed = 42,
    de_method = 't-test', resource_name = 'consensus',
    verbose = True, key_added = 'lr', n_jobs = 20
):
    adata.var['.ugene'] = adata.var_names.tolist()
    if gene_symbol is not None: adata.var_names = adata.var[gene_symbol].tolist()
    else: 
        adata.var['symbol'] = [str(x) if str(x) != 'nan' else y for x, y in zip(
            adata.var['gene'].tolist(), adata.var['ensembl'].tolist()
        )]

        adata.var.loc[adata.var['symbol'].isna(), 'symbol'] = \
            adata.var_names[adata.var['symbol'].isna()]
        adata.var_names = adata.var['symbol'].tolist()
        adata.var_names_make_unique()

    from exprmat.lr import flavors
    flavors[flavor](
        adata, taxa_source = taxa_source, taxa_dest = taxa_dest,
        groupby = groupby, use_raw = use_raw,
        min_cells = min_cells, expr_prop = expr_prop,
        # set to a smaller value. lianapy uses 1000 by default.
        n_perms = n_perms, seed = seed,
        de_method = de_method, resource_name = resource_name,
        verbose = verbose, key_added = key_added, n_jobs = n_jobs
    )

    adata.var_names = adata.var['.ugene'].tolist()


def rna_score_genes(
    adata, sample_name, taxa, gene_sets,
    identifier = 'uppercase', lognorm = 'X', random_state = 42,
    **kwargs
):
    from scanpy.tools import score_genes
    from exprmat.data.geneset import get_genesets, translate_id
    from exprmat.utils import choose_layer

    if isinstance(gene_sets, str):
        gs = get_genesets(taxa = taxa, name = gene_sets, identifier = identifier)
    else: gs = gene_sets

    genes = adata.var_names.tolist()
    genes = [x.replace('rna:', '') for x in genes]
    genes = translate_id(taxa, genes, 'ugene', identifier, keep_nones = True)

    mat = choose_layer(adata, layer = lognorm)
    temp = ad.AnnData(X = mat, var = adata.var)
    temp.var_names = [x if x is not None else 'na' for x in genes]
    temp.var_names_make_unique()

    # score genes
    for k in gs.keys():
        
        score_genes(
            temp, gs[k], score_name = 'score.' + k,  
            random_state = random_state, **kwargs
        )

        adata.obs['score.' + k] = temp.obs['score.' + k].tolist()
    
    return ['score.' + k for k in gs.keys()]


def rna_score_genes_gsva(
    adata, sample_name, taxa, gene_sets,
    identifier = 'uppercase', lognorm = 'X', random_state = 42,
    n_cores = 1, kcdf = 'Gaussian', weight = 1, min_genes = 15, max_genes = 1000, 
    append_to_obs = False, **kwargs
):
    from scanpy.tools import score_genes
    from exprmat.data.geneset import get_genesets, translate_id
    from exprmat.utils import choose_layer

    # per sample gsva
    PER_SAMPLE = False

    if PER_SAMPLE:

        samples = adata.obs['sample'].unique().tolist()
        for samp in samples:

            tmp = adata[adata.obs['sample'] == samp, :].copy()
            cellnames = tmp.obs_names.tolist()
            df = rna_gsva(
                tmp, sample_name, taxa = taxa, identifier = identifier, gene_sets = gene_sets,
                lognorm = lognorm, n_cores = n_cores, kcdf = kcdf,
                weight = weight, min_genes = min_genes, max_genes = max_genes
            )

            matrix = df.X

            if append_to_obs:
                # score genes
                gsets = df.var_names.tolist()
                for i, k in enumerate(gsets):
                    if ('nes.' + k) not in adata.obs.keys():
                        adata.obs['nes.' + k] = 0
                    adata.obs.loc[cellnames, 'nes.' + k] = matrix[:, i].T.tolist()

                return ['nes.' + k for k in gsets]
            else: return df
        
    else:

        df = rna_gsva(
            adata, sample_name, taxa = taxa, identifier = identifier, gene_sets = gene_sets,
            lognorm = lognorm, n_cores = n_cores, kcdf = kcdf,
            weight = weight, min_genes = min_genes, max_genes = max_genes
        )

        matrix = df.X

        if append_to_obs:
            # score genes
            gsets = df.var_names.tolist()
            for i, k in enumerate(gsets):
                adata.obs['nes.' + k] = matrix[:, i].T.tolist()
            return ['nes.' + k for k in gsets]
        
        else: return df


def rna_velocity(
    adata, sample_name, 
    neighbor_key: str = 'neighbors', neighbor_connectivity: str = 'connectivities', 
    n_neighbors: int = 35, hvg: str = 'vst.norm', velocity_key: str = 'velocity', 
    n_cpus = None, kwargs_filter = {}, kwargs_velocity = {}, 
    kwargs_velocity_graph = {}, kwargs_terminal_state = {}, 
    kwargs_pseudotime = { 'save_diffmap': True }
):
    from exprmat.dynamics import run_velocity
    run_velocity(
        adata, neighbor_key = neighbor_key, neighbor_connectivity = neighbor_connectivity,
        n_neighbors = n_neighbors, hvg = hvg, velocity_key = velocity_key,
        n_cpus = n_cpus, kwargs_filter = kwargs_filter, 
        kwargs_pseudotime = kwargs_pseudotime, kwargs_velocity = kwargs_velocity,
        kwargs_velocity_graph = kwargs_velocity_graph, kwargs_terminal_state = kwargs_terminal_state,
    )


def rna_consensus_nmf(
    adata, sample_name, 
    ks,
    counts: str = 'counts', tpm: str = 'norm', 
    hvg: str = 'vst.hvg', 
    min_counts: int = 3, 
    alpha_usage: float = 0, alpha_spectra: float = 0, 
    init: str = 'random', 
    max_nmf_iter: int = 1000, n_permutation: int = 100, 
    seed: int = 42, beta_loss: str = 'frobenius', 
    density_threshold: float = 0.5,
    local_neighborhood_size: float = 0.3,
    refit: bool = True,
    normalize_tpm_spectra: bool = False,
    key_added = 'cnmf',
    ncpus = 1
):
    from exprmat.clustering.cnmf import cnmf
    from exprmat.preprocessing import normalize

    if not tpm in adata.layers.keys():
        normalize(adata, counts = counts, dest = tpm, method = 'total')

    kwargs = {
        'counts': counts, 'tpm': tpm, 'hvg': hvg,
        'min_counts': min_counts,
        'alpha_usage': alpha_usage, 'alpha_spectra': alpha_spectra,
        'init': init,
        'max_nmf_iter': max_nmf_iter, 'n_iter': n_permutation,
        'seed': seed, 'beta_loss': beta_loss,
        'ncpus': ncpus, 'density_threshold': density_threshold,
        'local_neighborhood_size': local_neighborhood_size,
        'refit': refit, 'normalize_tpm_spectra': normalize_tpm_spectra,
        'ks': ks
    }

    comb, stats = cnmf(
        adata, comb = None,
        return_k = None, # for the first round, return statistics
        **kwargs
    )
    
    combs = {}
    for ky in comb.keys():
        combs[str(ky)] = comb[ky]

    stats = pd.DataFrame(stats, index = ['ncomps', 'threshold', 'silhoutte', 'error']).T
    adata.uns[key_added] = combs
    adata.uns[f'{key_added}.stats'] = stats
    adata.uns[f'{key_added}.args'] = kwargs


def rna_consensus_nmf_extract_k(
    adata, sample_name, k, nmf_slot = 'cnmf', 
    usage_added = 'cnmf.{0}', 
    spectra_added = 'cnmf.{0}',
    coef_added = 'cnmf.coef.{0}',
    spectra_cluster_dist_added = 'cnmf.dist.{0}',
    density_added = 'cnmf.density.{0}',
    **kwargs
):
    from exprmat.clustering.cnmf import cnmf
    from exprmat.preprocessing import normalize

    kwargs.update(adata.uns[f'{nmf_slot}.args'])
    
    comb = {}
    for ky in adata.uns[nmf_slot].keys():
        comb[int(ky)] = adata.uns[nmf_slot][ky]

    _, res = cnmf(
        adata, comb = comb,
        return_k = k, # extract specific k
        **kwargs
    )

    local_density, dist, _, rf_usages, spectra_tpm, usage_coef, _ = res
    rf_usages.columns = ['C' + str(x) for x in rf_usages.columns]
    adata.obsm[usage_added.format(k)] = rf_usages
    destindex = adata.var_names.tolist()

    adata.varm[spectra_added.format(k)] = np.zeros((adata.n_vars, len(spectra_tpm)))
    srcindex = spectra_tpm.columns.tolist()
    mapping = [destindex.index(x) for x in srcindex]
    adata.varm[spectra_added.format(k)][mapping, :] = spectra_tpm.T
    
    adata.varm[coef_added.format(k)] =  np.zeros((adata.n_vars, len(usage_coef)))
    srcindex = usage_coef.columns.tolist()
    mapping = [destindex.index(x) for x in srcindex]
    adata.varm[coef_added.format(k)][mapping, :] = usage_coef.T

    adata.uns[spectra_cluster_dist_added.format(k)] = dist
    adata.uns[density_added.format(k)] = local_density
    return


def rna_impute_magic(
    adata, sample_name, key_added = 'magic', solver = 'approximate', t = 'auto', 
    random_state = 42, n_jobs = 1, n_pca = 30, knn = 15,
    **kwargs
):
    from exprmat.preprocessing.impute.magic import magic
    imputer = magic(
        knn = knn, n_pca = n_pca, solver = solver, t = t, n_jobs = n_jobs, 
        random_state = random_state
    )

    imp = imputer.fit_transform(adata)
    adata.layers[key_added] = imp.X
    if hasattr(imputer, 'errors'): adata.uns['magic.errors'] = getattr(imputer, 'errors')
    if hasattr(imputer, 't_opt'): adata.uns['magic.t'] = getattr(imputer, 't_opt')


def atac_filter_cells(
    adata, sample_name, min_counts = 5000, max_counts = 100000, 
    min_tsse = 10, max_tsse = 100
):
    from exprmat.reader.peaks import filter_cells
    filter_cells(
        adata, min_counts = min_counts, max_counts = max_counts, 
        min_tsse = min_tsse, max_tsse = max_tsse
    )


def atac_make_bins(adata, sample_name, **kwargs):
    from exprmat.reader.peaks import add_tile_matrix
    add_tile_matrix(adata, **kwargs)

    # rename the chromosomal location
    chromosome = [x.split(':')[0] for x in adata.var_names.tolist()]
    start = [x.split(':')[1].split('-')[0] for x in adata.var_names.tolist()]
    end = [x.split(':')[1].split('-')[1] for x in adata.var_names.tolist()]
    adata.var['.seqid'] = chromosome
    adata.var['.start'] = start
    adata.var['.end'] = end
    adata.var['location'] = adata.var_names.tolist()

    i_taxa = cfg['taxa.reference'][adata.uns['assembly']]
    adata.var['unique'] = ('bin:' + i_taxa + ':') + adata.var_names
    adata.var_names = adata.var['unique'].tolist()
    

def atac_select_features(adata, sample_name, **kwargs):
    from exprmat.preprocessing.filter import select_features_atac_bins
    select_features_atac_bins(adata, **kwargs)


def atac_scrublet(adata, sample_name, scrublet_init_args = {}, scrublet_args = {}):
    from exprmat.preprocessing.doublets.scrublet import scrublet_init, scrublet_spectrum
    scrublet_init(adata, **scrublet_init_args)
    scrublet_spectrum(adata, **scrublet_args)


def atac_spectral(adata, sample_name, key_added = 'spectral', **kwargs):
    from exprmat.reduction.spectral import spectral
    spectral(adata, key_added = key_added, **kwargs)


def atac_infer_gene_activity(adata, sample_name, make_gene_args = {}, exact = True, **kwargs):
    
    if 'bedgraph' in adata.obs.keys():
        
        # query bedgraph values with ranges defined in peaks
        # and create a new modality named 'atac.p' (atac represent bin matrix,
        # atac.g represent gene matrix, and atac.p as peak matrix.)
        
        from exprmat.data.finders import get_genome_model
        gmodel = get_genome_model(adata.uns['assembly'])
        gmodel = gmodel.loc[gmodel['type'] == 'gene', :].copy()
        bedgraphs = adata.obs['bedgraph'].tolist()
        import pyBigWig as pybw
        import pyBedGraph as pybdg
        from rich.progress import track
        
        # generate a matrix with n_peaks * n_samples
        # obs tables are copied from obs, and var tables from peak table.

        mat = np.zeros(shape = (adata.n_obs, len(gmodel)), dtype = np.float32)
        for i, fp in enumerate(bedgraphs):

            ftype = 'bedgraph'
            fhandle = None
            if os.path.isfile(fp.replace('.bdg', '.bigwig')):
                fhandle = pybw.open(fp.replace('.bdg', '.bigwig'))
                ftype = 'bigwig'
            else: fhandle = pybdg.BedGraph('', fp)
            
            for i_region in track(range(len(gmodel)), description = f'querying genes for {fp}'):

                peak = gmodel.iloc[i_region, :]
                # retrieve statistics.
                if ftype == 'bigwig':
                    # open the companion bigwig file for more efficient io.
                    mat[i, i_region] = (peak['end'] - peak['start']) * fhandle.stats(
                        peak['ucsc'], peak['start'], peak['end'], type = 'mean', exact = exact
                    )[0]

                else: raise NotImplementedError
        
        from scipy.sparse import csr_matrix
        gdata = ad.AnnData(X = csr_matrix(np.nan_to_num(mat)), obs = adata.obs)
        gdata.var_names = gmodel['gid'].tolist()

        taxa = cfg['taxa.reference'][adata.uns['assembly']]
        from exprmat.utils import genes_from_names
        gmask, names, _ = genes_from_names(gdata.var_names.tolist(), taxa)
        gdata = gdata[:, gmask].copy()
        gdata.var_names = names
        gdata.var = search_genes(gdata.var_names.tolist())
        gdata = gdata[:, ~gdata.var_names.duplicated()].copy()
        gdata.uns['assembly'] = adata.uns['assembly']
        gdata.uns['assembly.size'] = adata.uns['assembly.size']
        return gdata
    
    else:
        from exprmat.reader.peaks import make_gene_matrix
        taxa = cfg['taxa.reference'][adata.uns['assembly']]
        # this method requires the location as variable names
        adata.var_names = adata.var['location'].tolist()
        gene_activity = make_gene_matrix(adata, adata.uns['assembly'], **make_gene_args)
        adata.var_names = adata.var['unique'].tolist()

        gene_activity.X = gene_activity.X.astype(np.float32)
        # obs slots are copied from the original bins adata.
        for obsmk in adata.obsm.keys():
            if obsmk in ['paired', 'single']: continue
            gene_activity.obsm[obsmk] = adata.obsm[obsmk]
        # variable names are gene names
        from exprmat.utils import genes_from_names
        gmask, names, _ = genes_from_names(gene_activity.var_names.tolist(), taxa)
        gene_activity = gene_activity[:, gmask].copy()
        gene_activity.var_names = names
        gene_activity.var = search_genes(gene_activity.var_names.tolist())

        # remove non-expressing genes.
        gmask = gene_activity.X.sum(axis = 0) > 0.01
        gene_activity = gene_activity[:, gmask.tolist()[0]].copy()
        gene_activity.uns['assembly'] = adata.uns['assembly']
        gene_activity.uns['assembly.size'] = adata.uns['assembly.size']
        return gene_activity


def atac_call_peaks(adata, sample_name, key_added = 'peaks', **kwargs):
    from exprmat.peaks.callpeak import (
        call_peak_from_bedgraph, 
        call_peak_from_fragments
    )

    if 'bedgraph' in adata.obs.keys():
        if key_added not in adata.uns.keys():
            adata.uns[key_added] = {}
        
        for bdg, samp in zip(adata.obs['bedgraph'].tolist(), adata.obs['sample'].tolist()):
            if len(bdg) == 0: continue
            adata.uns[key_added][samp] = call_peak_from_bedgraph(bdg, **kwargs)
    
    else: call_peak_from_fragments(adata, key_added = key_added, **kwargs)


def atac_merge_peaks(adata, sample_name, key_peaks = 'peaks', key_added = 'peaks.merged', key_groups = 'peaks.group', groupby = 'group', **kwargs):
    
    from exprmat.peaks.idr.idr import idr
    if 'bedgraph' in adata.obs.keys():
        sample_key = list(adata.uns[key_peaks].keys())
        group_to_sample_mapping = defaultdict(list)
        for g, k in zip(adata.obs[groupby], adata.obs['sample']):
            group_to_sample_mapping[g] += [k]
        
        # merge within duplicates
        group_peaks = {}
        for g in group_to_sample_mapping.keys():
            
            if len(group_to_sample_mapping[g]) == 1:
                group_peaks[g] = adata.uns[key_peaks][group_to_sample_mapping[g][0]]
            
            elif len(group_to_sample_mapping[g]) == 2:
                # info(f'merging peaks from {group_to_sample_mapping[g][0]} and {group_to_sample_mapping[g][1]} (group {g})')
                group_peaks[g] = idr(
                    df1 = adata.uns[key_peaks][group_to_sample_mapping[g][0]],
                    df2 = adata.uns[key_peaks][group_to_sample_mapping[g][1]],
                    signal_type = 'score', summit_type = 'summit', peak_merge_fn = np.sum,
                    use_nonoverlapping_peaks = False, **kwargs
                )

            else:
                from exprmat.utils import reduce
                lst = [adata.uns[key_peaks][x] for x in group_to_sample_mapping[g]]
                group_peaks[g] = reduce(
                    idr, lst, 
                    signal_type = 'score', summit_type = 'summit', peak_merge_fn = np.sum,
                    use_nonoverlapping_peaks = False, **kwargs
                )
            
            info(f'yield {len(group_peaks[g])} peaks for group {g}')

        adata.uns[key_groups] = group_peaks

        # merge between groups (nonredundant)
        if len(group_peaks) == 1:
            adata.uns[key_added] = group_peaks[list(group_peaks.keys())[0]]
        elif len(group_peaks) == 2:
            keys = list(group_peaks.keys())
            adata.uns[key_added] = idr(
                df1 = group_peaks[keys[0]],
                df2 = group_peaks[keys[1]],
                signal_type = 'score', summit_type = 'summit', peak_merge_fn = np.sum,
                use_nonoverlapping_peaks = True,
                only_merge_peaks = True, **kwargs
            )
        else:
            from exprmat.utils import reduce
            lst = [group_peaks[x] for x in group_peaks.keys()]
            adata.uns[key_added] = reduce(
                idr, lst,
                signal_type = 'score', summit_type = 'summit', peak_merge_fn = np.sum,
                use_nonoverlapping_peaks = True,
                only_merge_peaks = True, **kwargs
            )
    
    else: raise NotImplementedError


def atac_make_peak_matrix(adata, sample_name, key_peaks = 'peaks.merged', exact = True, **kwargs):
    
    from exprmat.peaks.idr.idr import idr

    if 'bedgraph' in adata.obs.keys():
        
        # query bedgraph values with ranges defined in peaks
        # and create a new modality named 'atac.p' (atac represent bin matrix,
        # atac.g represent gene matrix, and atac.p as peak matrix.)
        
        peak_table = adata.uns[key_peaks]
        peak_table.index = (
            'peak:' + peak_table['chr'] + ':' + peak_table['start'].astype('str') + 
            '-' + peak_table['end'].astype('str')
        )

        peak_table.index.name = None
        bedgraphs = adata.obs['bedgraph'].tolist()
        
        import pyBigWig as pybw
        import pyBedGraph as pybdg
        from rich.progress import track
        
        # generate a matrix with n_peaks * n_samples
        # obs tables are copied from obs, and var tables from peak table.

        mat = np.zeros(shape = (adata.n_obs, len(peak_table)), dtype = np.float32)
        for i, fp in enumerate(bedgraphs):

            ftype = 'bedgraph'
            fhandle = None
            if os.path.isfile(fp.replace('.bdg', '.bigwig')):
                fhandle = pybw.open(fp.replace('.bdg', '.bigwig'))
                ftype = 'bigwig'
            else: fhandle = pybdg.BedGraph('', fp)
            
            for i_region in track(range(len(peak_table)), description = f'querying peaks for {fp}'):

                peak = peak_table.iloc[i_region, :]
                # retrieve statistics.
                if ftype == 'bigwig':
                    # open the companion bigwig file for more efficient io.
                    mat[i, i_region] = (peak['end'] - peak['start']) * fhandle.stats(
                        peak['chr'], peak['start'], peak['end'], type = 'mean', exact = exact
                    )[0]

                else: raise NotImplementedError

        from scipy.sparse import csr_matrix
        pdata = ad.AnnData(X = csr_matrix(np.nan_to_num(mat)), obs = adata.obs, var = peak_table)
        pdata.uns['assembly'] = adata.uns['assembly']
        pdata.uns['assembly.size'] = adata.uns['assembly.size']
        return pdata
    
    else: raise NotImplementedError


def atacp_annotate_peak(
    adata, sample_name, 
    annotation_key = 'type', 
    gene_key = 'tss.nearest',
    ugene_key = 'ugene',
    gname_key = 'gene', 
    distance_key = 'tss.dist'
):

    # annotate peaks onto nearest gene features.
    from genomicranges.GenomicRanges import GenomicRanges as granges
    from exprmat.data.finders import (
        get_genome_promoters,
        get_genome_utr3,
        get_genome_utr5,
        get_genome_first_exonic,
        get_genome_other_exonic,
        get_genome_first_intronic,
        get_genome_transcript
    )

    query = granges.from_pandas(pd.DataFrame({
        "seqnames": adata.var['chr'].tolist(), 
        "starts": adata.var['start'].tolist(), 
        "ends": adata.var['end'].tolist(), 
        "strand": [x if x in ['+', '-'] else '*' for x in adata.var['strand'].tolist()]
    }))

    adata.var[annotation_key] = 'intergenic'
    adata.var[gene_key] = '.'
    adata.var[ugene_key] = '.'
    adata.var[gname_key] = '.'
    adata.var[distance_key] = float('nan')

    def search_and_apply(adata, query: granges, search: granges, annotation):
        result = search.find_overlaps(query, query_type = 'any', select = 'first')
        adata.var.iloc[result['query_hits'], adata.var.columns.tolist().index(annotation_key)] = annotation
    
    promoters = get_genome_promoters(adata.uns['assembly'], True)
    search_and_apply(adata, query, get_genome_transcript(adata.uns['assembly'], True), 'genebody')
    search_and_apply(adata, query, get_genome_first_intronic(adata.uns['assembly'], True), 'first-intron')
    search_and_apply(adata, query, get_genome_other_exonic(adata.uns['assembly'], True), 'other-exons')
    search_and_apply(adata, query, get_genome_first_exonic(adata.uns['assembly'], True), 'first-exon')
    search_and_apply(adata, query, get_genome_utr3(adata.uns['assembly'], True), 'utr3')
    search_and_apply(adata, query, get_genome_utr5(adata.uns['assembly'], True), 'utr5')
    search_and_apply(adata, query, promoters, 'promoter')

    tss = promoters.narrow(start = 2999, end = 2999)
    nearest_gene = tss.nearest(query, select = 'arbitrary', ignore_strand = False)
    adata.var[gene_key] = tss.mcols[nearest_gene, 'gene']['gene']
    peak_index = adata.var['start'] + adata.var['summit']
    central_tss = (tss[nearest_gene.tolist()].start + tss[nearest_gene.tolist()].end) / 2
    adata.var[distance_key] = peak_index - central_tss

    taxa = cfg['taxa.reference'][adata.uns['assembly']]
    from exprmat.utils import genes_from_names
    gmask, names, _ = genes_from_names(adata.var[gene_key].tolist(), taxa)
    adata.var.loc[gmask, ugene_key] = names
    gmeta = search_genes(names)
    adata.var.loc[gmask, gname_key] = gmeta['gene'].tolist()
    pass


def atacp_retrieve_sequence(adata, sample_name):
    from exprmat.peaks.sequence import query_sequence
    query_sequence(adata)
    

def rna_plot_qc(adata, sample_name, **kwargs):
    from exprmat.preprocessing.plot import rna_plot_qc_metrics
    return rna_plot_qc_metrics(adata, sample_name, **kwargs)


def rna_plot_embedding(adata, sample_name, **kwargs):
    from exprmat.reduction.plot import embedding
    return embedding(adata, sample_name = sample_name, **kwargs)


def rna_plot_embedding_mask(adata, sample_name, color, values, **kwargs):
    from exprmat.reduction.plot import embedding
    if color in adata.obs.keys():
        adata.obs['.mask'] = [x in values for x in adata.obs[color]]
    else: error('`color` must be within adata.obs')
    fig = embedding(adata, sample_name = sample_name, color = '.mask', **kwargs)
    if color in adata.obs.keys(): del adata.obs['.mask']
    return fig


def rna_plot_embedding_atlas(adata, sample_name, **kwargs):
    from exprmat.reduction.plot import embedding_atlas
    return embedding_atlas(adata, sample_name = sample_name, **kwargs)


def rna_plot_gene_gene(adata, sample_name, **kwargs):
    from exprmat.reduction.plot import gene_gene
    return gene_gene(adata, sample_name = sample_name, **kwargs)

def rna_plot_gene_gene_regress(adata, sample_name, **kwargs):
    from exprmat.reduction.plot import gene_gene_regress
    return gene_gene_regress(adata, sample_name = sample_name, **kwargs)


def rna_plot_markers(adata, sample_name, figsize, dpi, **kwargs):
    from exprmat.plotting.de import marker_plot
    pl = marker_plot(adata, sample_name = sample_name, **kwargs)

    from scanpy.plotting import DotPlot, StackedViolin, MatrixPlot
    if isinstance(pl, DotPlot) or isinstance(pl, StackedViolin) or isinstance(pl, MatrixPlot):
        pl.width = figsize[0]
        pl.height = figsize[1]
        pl.show()
        pl.fig.set_dpi(dpi)
        return pl.fig
    
    else:
        fig = pl['heatmap_ax'].figure if 'heatmap_ax' in pl.keys() else None
        fig = pl['track_axes'][0].figure if 'track_axes' in pl.keys() else fig

        if fig:
            fig.set_dpi(dpi)
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])
            return fig
    


def rna_plot_expression_bar(
    adata, sample_name, gene, group, split = None,
    slot = 'X', selected_groups = None, selected_splits = None, palette = ['red', 'black'], 
    figsize = (6,3), dpi = 100, **kwargs
):
    from exprmat.plotting.expression import barplot
    pl = barplot(
        adata, gene = gene, slot = slot, group = group,
        split = split, selected_groups = selected_groups, 
        selected_splits = selected_splits, palette = palette,
        size = figsize, dpi = dpi, **kwargs
    )
    return pl


def rna_plot_expression_bar_multiple(
    adata, sample_name, features, ncols, group, split = None,
    slot = 'X', selected_groups = None, palette = ['red', 'black'], 
    figsize = (6,3), dpi = 100, **kwargs
):
    from exprmat.plotting.expression import barplot
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    n_features = len(features)
    nrows = n_features // ncols
    if n_features % ncols != 0: nrows += 1
    fig, axes = plt.subplots(nrows, ncols, dpi = dpi)

    for feat_id in range(len(features)):
        try:
            if len(axes.shape) == 2:
                barplot(
                    adata, gene = features[feat_id], slot = slot, group = group,
                    ax = axes[feat_id // ncols, feat_id % ncols],
                    split = split, selected_groups = selected_groups, palette = palette,
                    size = figsize, dpi = dpi, **kwargs
                )

            elif len(axes.shape) == 1:
                barplot(
                    adata, gene = features[feat_id], slot = slot, group = group,
                    ax = axes[feat_id],
                    split = split, selected_groups = selected_groups, palette = palette,
                    size = figsize, dpi = dpi, **kwargs
                )
        except: pass
    
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    return fig


def rna_plot_compare_scatter(
    adata, sample_name, group_x, group_y, key, 
    slot = 'X', markers = [], 
    figsize = (4, 4), dpi = 100
):
    from exprmat.plotting.expression import compare_scatter
    return compare_scatter(
        adata, group_x = group_x, group_y = group_y,
        key = key, slot = slot, markers = markers, sample = sample_name,
        figsize = figsize, dpi = dpi
    )


def rna_plot_proportion(
    adata, sample_name, major, minor, plot = 'bar', cmap = 'turbo',
    normalize = 'columns', figsize = (5,3), stacked = False, wedge = 0.4
):
    if normalize == 'major': normalize = 'columns'
    if normalize == 'minor': normalize = 'index'
    tmp = pd.crosstab(adata.obs[major], adata.obs[minor], normalize = normalize)

    def get_palette(n):
        if n + '.colors' in adata.uns.keys():
            return adata.uns[n + '.colors']
        else: return 'turbo'

    if plot == 'bar':
        fig = tmp.plot.bar(stacked = stacked, figsize = figsize, grid = False, cmap = cmap)
        fig.legend(loc = None, bbox_to_anchor = (1, 1), frameon = False)
        fig.set_ylabel(f'Proportion ({minor})')
        fig.set_xlabel(major)
        fig.spines['right'].set_visible(False)
        fig.spines['top'].set_visible(False)
        if normalize == 'index':
            fig.set_ylim(0, 1)
        fig.figure.tight_layout()
        return fig.figure
    
    elif plot == 'pie':
        fig = tmp.plot.pie(
            subplots = True,
            radius = 0.8, autopct = '%1.1f%%'
            # colors = get_palette(major)
        )
        for x in fig: x.get_legend().remove()
        fig[0].figure.tight_layout()
        fig[0].figure.set_figwidth(figsize[0])
        fig[0].figure.set_figheight(figsize[1])
        return fig[0].figure
    
    elif plot == 'donut':
        fig = tmp.plot.pie(
            subplots = True, 
            wedgeprops = dict(width = wedge),
            radius = 0.8, autopct = '%1.1f%%',
            colors = get_palette(major)
        )
        for x in fig: x.get_legend().remove()
        fig[0].figure.tight_layout()
        fig[0].figure.set_figwidth(figsize[0])
        fig[0].figure.set_figheight(figsize[1])
        return fig[0].figure
    
    else: error('unsupported plotting type.')


def rna_plot_kde(
    adata, sample_name, basis, kde, grouping_key, 
    figsize, dpi, groups = None, ncols = 1, background = 'leiden',
    annotate = True, annotate_fontsize = 9, **kwargs
):
    from exprmat.reduction.plot import embedding
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    cats = adata.obs[grouping_key].cat.categories.tolist()
    if groups is None: groups = cats

    n_features = len(groups)
    nrows = n_features // ncols
    if n_features % ncols != 0: nrows += 1
    fig, axes = plt.subplots(nrows, ncols, dpi = dpi)
    
    for feat_id in range(len(groups)):
        try:
            subset = adata[adata.obs[grouping_key] == groups[feat_id],:]
            ann = subset.obs[background].value_counts().index.tolist()

            if len(axes.shape) == 2:

                # we need to plot a whole map as the background,
                # for the subset maybe extremely biased and uncomparable.
                embedding(
                    adata, basis, color = background,
                    default_color = '#eeeeee', ptsize = 8,
                    annotate = annotate, annotate_style = 'text',
                    legend = False, contour_plot = False,
                    annotate_fontsize = annotate_fontsize,
                    annotate_only = ann,
                    cmap = None, add_outline = True, outline_color = '#777777',
                    ax = axes[feat_id // ncols, feat_id % ncols], dpi = dpi,
                    sample_name = sample_name, title = groups[feat_id]
                )
            
                embedding(
                    subset, 
                    basis, color = kde,
                    ax = axes[feat_id // ncols, feat_id % ncols], dpi = dpi,
                    sample_name = sample_name, title = groups[feat_id], **kwargs
                )

            elif len(axes.shape) == 1:

                embedding(
                    adata, basis, color = background,
                    default_color = '#eeeeee', ptsize = 8,
                    annotate = annotate, annotate_style = 'text',
                    legend = False, contour_plot = False,
                    annotate_fontsize = annotate_fontsize,
                    annotate_only = ann,
                    cmap = None, add_outline = True, outline_color = '#777777',
                    ax = axes[feat_id], dpi = dpi,
                    sample_name = sample_name, title = groups[feat_id]
                )

                embedding(
                    subset, 
                    basis, color = kde,
                    ax = axes[feat_id], dpi = dpi,
                    sample_name = sample_name, title = groups[feat_id], **kwargs
                )
                
        except: pass
    
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    return fig


def rna_plot_dot(
    adata, sample_name, figsize, dpi, 
    var_names, groupby, *, log = False,
    categories_order = None, expression_cutoff = 0.0, dendrogram = False,
    dendrogram_rep = 'pca',
    mean_only_expressed = False, standard_scale = 'var', 
    title = None, colorbar_title = 'Mean expression', 
    size_title = 'Fraction of cells (%)', gene_symbols = 'gene', 
    var_group_positions = None, var_group_labels = None, 
    var_group_rotation = None, layer = None, swap_axes = False, 
    dot_color_df = None, vmin = None, vmax = None, vcenter = None, norm = None, 
    cmap = 'turbo', dot_max = None, dot_min = None, smallest_dot = 0.0, **kwds
):
    from scanpy.plotting import dotplot
    from scanpy.tools import dendrogram as dend

    if dendrogram:
        dend(
            adata, groupby = groupby, use_rep = dendrogram_rep, 
            optimal_ordering = True, key_added = f'dendrogram.{groupby}'
        )

    pl = dotplot(
        adata, var_names, groupby = groupby, figsize = figsize,
        log = log, return_fig = True, dendrogram = f'dendrogram.{groupby}' if dendrogram else None,
        categories_order = categories_order, expression_cutoff = expression_cutoff, 
        mean_only_expressed = mean_only_expressed, standard_scale = standard_scale, 
        title = title, colorbar_title = colorbar_title, 
        size_title = size_title, gene_symbols = gene_symbols, 
        var_group_positions = var_group_positions, var_group_labels = var_group_labels, 
        var_group_rotation = var_group_rotation, layer = layer, swap_axes = swap_axes, 
        dot_color_df = dot_color_df, vmin = vmin, vmax = vmax, vcenter = vcenter, norm = norm, 
        cmap = cmap, dot_max = dot_max, dot_min = dot_min, smallest_dot = smallest_dot, **kwds
    )

    pl.show()
    pl.fig.set_dpi(dpi)
    return pl.fig


def rna_plot_heatmap(
    adata, sample_name, var_names, groupby, dpi = 100, categories_order = None,
    cmap = 'turbo', figsize = (4, 8), standard_scale = None, var_identifier = 'gene',
    var_group_labels = None, show_gene_labels = None
):
    from scanpy.plotting import heatmap
    from exprmat.data.geneset import translate_id
    from exprmat.utils import translate_variables

    if var_identifier is None:
        if isinstance(var_names, dict):
            for k in var_names.keys():
                var_names[k] = translate_variables(adata, var_names[k])

        elif isinstance(var_names, list):
            var_names = translate_variables(adata, var_names)

    if (groupby is not None) and (categories_order is not None):
        subset_adata = [x in categories_order for x in adata.obs[groupby].tolist()]
        if np.sum(np.array(subset_adata)) < len(subset_adata):
            subset_adata = adata[subset_adata, :]
        else: subset_adata = adata
    else: subset_adata = adata

    pl = heatmap(
        adata = subset_adata, var_names = var_names, groupby = groupby, swap_axes = False,
        cmap = cmap, show = False, figsize = figsize, standard_scale = standard_scale,
        gene_symbols = var_identifier, var_group_labels = var_group_labels,
        var_group_rotation = 90, show_gene_labels = show_gene_labels
        # categories_order = categories_order
    )

    pl['heatmap_ax'].figure.set_dpi(dpi)
    return pl['heatmap_ax'].figure


def adata_plot_matrix(
    adata, sample_name, layer = 'X', obs_names = None, var_names = None,
    figsize = (3, 3), ax = None, **kwargs
):
    from exprmat.plotting.expression import matrix
    return matrix(
        adata, layer = layer, obs_names = obs_names, var_names = var_names,
        figsize = figsize, ax = ax, **kwargs
    )


def rna_plot_multiple_embedding(
    adata, sample_name, basis, features, ncols, 
    figsize = (3, 3), dpi = 100, **kwargs
):
    from exprmat.reduction.plot import embedding
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    n_features = len(features)
    nrows = n_features // ncols
    if n_features % ncols != 0: nrows += 1
    fig, axes = plt.subplots(nrows, ncols, dpi = dpi)

    for feat_id in range(len(features)):
        try:
            if len(axes.shape) == 2:
                embedding(
                    adata, basis, color = features[feat_id],
                    ax = axes[feat_id // ncols, feat_id % ncols],
                    sample_name = sample_name, dpi = dpi, **kwargs
                )

            elif len(axes.shape) == 1:
                embedding(
                    adata, basis, color = features[feat_id],
                    ax = axes[feat_id], dpi = dpi,
                    sample_name = sample_name, **kwargs
                )
        except: pass
    
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    return fig


def rna_plot_multiple_embedding_atlas(
    adata, sample_name, basis, features, ncols, 
    figsize = (3, 3), dpi = 100, **kwargs
):
    from exprmat.reduction.plot import embedding_atlas
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    n_features = len(features)
    nrows = n_features // ncols
    if n_features % ncols != 0: nrows += 1
    fig, axes = plt.subplots(nrows, ncols, dpi = dpi)

    for feat_id in range(len(features)):
        try:
            if len(axes.shape) == 2:
                embedding_atlas(
                    adata, basis, color = features[feat_id],
                    ax = axes[feat_id // ncols, feat_id % ncols],
                    sample_name = sample_name, dpi = dpi, **kwargs
                )

            elif len(axes.shape) == 1:
                embedding_atlas(
                    adata, basis, color = features[feat_id],
                    ax = axes[feat_id], dpi = dpi,
                    sample_name = sample_name, **kwargs
                )
        except: pass
    
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    return fig


def rna_plot_multiple_gene_gene(
    adata, sample_name, color, features_xy, ncols, 
    figsize = (3, 3), **kwargs
):
    from exprmat.reduction.plot import gene_gene
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')

    n_features = len(features_xy)
    nrows = n_features // ncols
    if n_features % ncols != 0: nrows += 1
    fig, axes = plt.subplots(nrows, ncols)

    for feat_id in range(len(features_xy)):
        try:
            if len(axes.shape) == 2:
                gene_gene(
                    adata, 
                    gene_x = features_xy[feat_id][0],
                    gene_y = features_xy[feat_id][1],
                    color = color,
                    ax = axes[feat_id // ncols, feat_id % ncols],
                    sample_name = sample_name, **kwargs
                )

            elif len(axes.shape) == 1:
                gene_gene(
                    adata, 
                    gene_x = features_xy[feat_id][0],
                    gene_y = features_xy[feat_id][1],
                    color = color,
                    ax = axes[feat_id],
                    sample_name = sample_name, **kwargs
                )
        except: pass
    
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    return fig


def rna_plot_cnv_matrix(adata, sample_name, **kwargs):
    from exprmat.plotting.cnv import chromosome_heatmap
    return chromosome_heatmap(adata, sample_name = sample_name, **kwargs)


def rna_plot_gsea_running_es(
    adata, sample_name, gsea, terms, figsize = (4, 4), colors = None, title = None, **kwargs
):
    from exprmat.plotting.gse import esplot
    return esplot(
        adata, sample_name = sample_name, title = title,
        gsea = gsea, terms = terms, figsize = figsize, colors = colors, **kwargs
    )


def rna_plot_gsea_dotplot(
    adata, sample_name, gsea_key, max_fdr = 1, max_p = 0.05, top_term: int = 100,
    colour = 'p', title = "", cmap = 'turbo', figsize = (3, 2), cutoff = 1, ptsize = 5,
    terms = None, formatter = lambda x: x
):
    from exprmat.plotting.gse import gsea_dotplot
    return gsea_dotplot(
        rna_get_gsea(adata, None, gsea_key, max_fdr = max_fdr, max_p = max_p),
        column = colour, x = 'nes', y = 'name', title = gsea_key if title is None else title,
        cmap = cmap, size = ptsize, figsize = figsize, cutoff = cutoff, top_term = top_term,
        terms = terms, formatter = formatter
    )


def rna_plot_opa_dotplot(
    adata, sample_name, opa_key, max_fdr = 1, max_p = 0.05, top_term: int = 100, terms = None,
    colour = 'fdr', title = None, cmap = 'turbo', figsize = (3, 2), cutoff = 1, ptsize = 5,
    formatter = lambda x: x
):
    from exprmat.plotting.gse import opa_dotplot
    return opa_dotplot(
        rna_get_opa(adata, None, opa_key, max_fdr = max_fdr, max_p = max_p),
        column = colour, x = 'or', y = 'term', title = opa_key if title is None else title,
        cmap = cmap, size = ptsize, figsize = figsize, cutoff = cutoff, top_term = top_term,
        terms = terms, formatter = formatter
    )


def rna_plot_lr_dotplot(adata, sample_name, lr_key, uns_key = None, **kwargs):
    from exprmat.plotting.lr import lr_dotplot
    return lr_dotplot(adata = adata, uns_key = lr_key, **kwargs)


def rna_plot_lr_circleplot(adata, sample_name, lr_key, uns_key = None, **kwargs):
    from exprmat.plotting.lr import circleplot
    return circleplot(adata = adata, uns_key = lr_key, **kwargs)


def rna_plot_lr_heatmap(adata, sample_name, lr_key, uns_key = None, **kwargs):
    from exprmat.plotting.lr import heatmap
    return heatmap(adata = adata, uns_key = lr_key, **kwargs)


def rna_plot_volcano(
    adata, sample_name, de_slot = 'deg', label = [],
    show_all = False, min_pct = 0, max_pct_reference = 1, 
    min_lfc = -25, max_lfc = 25, remove_zero_pval = False,
    highlight_min_logp = 5, highlight_min_lfc = 1.5,
    xlim = 5, ylim = 100,
    figsize = (3, 3), dpi = 100, **kwargs):
    from exprmat.plotting.expression import volcano
    return volcano(
        adata = adata, de_slot = de_slot, label = label,
        show_all = show_all, min_pct = min_pct, max_pct_reference = max_pct_reference, 
        min_lfc = min_lfc, max_lfc = max_lfc, remove_zero_pval = remove_zero_pval,
        highlight_min_logp = highlight_min_logp, highlight_min_lfc = highlight_min_lfc,
        xlim = xlim, ylim = ylim,
        figsize = figsize, dpi = dpi, **kwargs
    )


def rna_plot_spliced_proportions(adata, sample_name, **kwargs):
    from exprmat.plotting.velocity import proportions
    return proportions(adata, **kwargs)


def rna_plot_velocity_streamline(
    adata, sample_name, basis: str = 'umap',
    vkey: str = "velocity",
    neighbor_key: str = 'neighbors',
    color = 'leiden', contour_plot = False, figsize = (5, 5), dpi = 100,
    density: int = 2, **kwargs
):
    from exprmat.plotting.velocity import velocity_embedding_stream
    return velocity_embedding_stream(
        adata, basis = basis, vkey = vkey, neighbor_key = neighbor_key, 
        color = color, contour_plot = contour_plot, figsize = figsize, dpi = dpi,
        density = density, **kwargs
    )


def rna_plot_velocity_gene(
    adata, sample_name, basis = 'umap', gene = None, groupby = 'cell.type',
    vkey = 'velocity', mode = "stochastic", neighbor_key = 'neighbors', 
    highly_variable = 'vst.hvg', figsize = (14, 28), dpi = 100, **kwargs
):
    from exprmat.plotting.velocity import velocity
    return velocity(
        adata, basis = basis, vkey = vkey, neighbor_key = neighbor_key, 
        gene = gene, groupby = groupby, mode = mode, highly_variable = highly_variable,
        figsize = figsize, dpi = dpi, **kwargs
    )


def adata_plot_sankey(adata, sample_name, obs1, obs2, exclude_values = ['na', 'nan'], **kwargs):
    from exprmat.plotting.sankey import sankey
    o1 = adata.obs[obs1].tolist()
    o2 = adata.obs[obs2].tolist()
    filters = [
        (ox1 not in exclude_values) and (ox2 not in exclude_values) 
        for ox1, ox2 in zip(o1, o2)
    ]

    return sankey(adata.obs.loc[filters, obs1], adata.obs.loc[filters, obs2], **kwargs)


def rna_plot_cnmf_density(adata, sample_name, **kwargs):
    from exprmat.plotting.cnmf import cnmf_density
    return cnmf_density(adata, **kwargs)


def rna_plot_cnmf_silhoutte(adata, sample_name, **kwargs):
    from exprmat.plotting.cnmf import cnmf_silhoutte
    return cnmf_silhoutte(adata, **kwargs)


def rna_plot_cnmf_distance_comps(adata, sample_name, **kwargs):
    from exprmat.plotting.cnmf import cnmf_distance_comps
    return cnmf_distance_comps(adata, **kwargs)


def rna_plot_cnmf_distance_usages(adata, sample_name, **kwargs):
    from exprmat.plotting.cnmf import cnmf_distance_usages
    return cnmf_distance_usages(adata, **kwargs)


def rna_plot_cnmf_distance_modules(adata, sample_name, **kwargs):
    from exprmat.plotting.cnmf import cnmf_distance_modules
    return cnmf_distance_modules(adata, **kwargs)


def atac_plot_qc(adata, sample_name, **kwargs):
    from exprmat.plotting.atac import atac_qc_metrics
    return atac_qc_metrics(adata, sample_name, **kwargs)


def rna_get_gsea(adata, sample_name, gsea_slot = 'gsea', max_fdr = 1.00, max_p = 0.05):
    
    df = {
        'name': [],
        'es': [],
        'nes': [],
        'p': [],
        'fwerp': [],
        'fdr': [],
        'tag': []
    }

    for gs in adata.uns[gsea_slot]['results'].keys():
        data = adata.uns[gsea_slot]['results'][gs]
        df['name'].append(gs)
        df['es'].append(data['es'])
        df['nes'].append(data['nes'])
        df['p'].append(data['p'])
        df['fwerp'].append(data['fwerp'])
        df['fdr'].append(data['fdr'])
        df['tag'].append(data['tag'])
    
    df = pd.DataFrame(df)
    if max_fdr is not None:
        df = df[df['fdr'] <= max_fdr]
    if max_p is not None:
        df = df[df['p'] <= max_p]
    
    df = df.sort_values(['fdr', 'p'])
    return df


def rna_get_opa(adata, sample_name, gsea_slot = 'gsea', max_fdr = 1.00, max_p = 0.05):
    
    df = pd.DataFrame(adata.uns[gsea_slot])

    if max_fdr is not None:
        df = df[df['fdr'] <= max_fdr]
    if max_p is not None:
        df = df[df['p'] <= max_p]
    
    df = df.sort_values(['fdr', 'p'])
    return df


def rna_get_lr(
    adata, sample_name, lr_slot = 'lr', source_labels = None, target_labels = None,
    ligand_complex = None, receptor_complex = None, 
    filter_fun = None, top_n: int = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
):
    
    from exprmat.plotting.lr import prepare_lr, filter_by, topn
    liana_res = prepare_lr(
        adata = adata,
        liana_res = None,
        source_labels = source_labels,
        target_labels = target_labels,
        ligand_complex = ligand_complex,
        receptor_complex = receptor_complex,
        uns_key = lr_slot
    )

    liana_res = filter_by(liana_res, filter_fun)
    liana_res = topn(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)
    return liana_res


def rna_get_markers(
    adata, sample_name, de_slot = 'markers', group_name = None, max_q = None,
    min_pct = 0.25, max_pct_reference = 0.75, min_lfc = 1, max_lfc = 100, remove_zero_pval = False
):
    params = adata.uns[de_slot]['params']

    # default value for convenience
    if len(adata.uns[de_slot]['differential']) == 1 and group_name == None:
        group_name = list(adata.uns[de_slot]['differential'].keys())[0]

    tab = adata.uns[de_slot]['differential'][group_name]

    if min_pct is not None and 'pct' in tab.columns:
        tab = tab[tab['pct'] >= min_pct]
    if max_pct_reference is not None and 'pct.reference' in tab.columns:
        tab = tab[tab['pct.reference'] <= max_pct_reference]
    if min_lfc is not None and 'lfc' in tab.columns:
        tab = tab[tab['lfc'] >= min_lfc]
    if max_lfc is not None and 'lfc' in tab.columns:
        tab = tab[tab['lfc'] <= max_lfc]
    if remove_zero_pval:
        tab = tab[~ np.isinf(tab['log10.q'].to_numpy())]
    if max_q is not None and 'q' in tab.columns:
        tab = tab[tab['q'] <= max_q]
    
    info(
        'fetched diff `' + red(group_name) + '` over `' + green(params['reference']) + '` ' + 
        f'({len(tab)} genes)'
    )
    tab = tab.sort_values(by = ['scores'], ascending = False)
    return tab


def rnaspc_transform(adata, sample, xfunc, yfunc):

    adata.obsm['spatial'][:, 0] = xfunc(adata.obsm['spatial'][:, 0])
    adata.obsm['spatial'][:, 1] = yfunc(adata.obsm['spatial'][:, 1])


def rnaspc_roi(adata, sample, spsample, xlim, ylim, sample_added):

    selection = (
        (adata.obsm['spatial'][:, 0] > xlim[0]) &
        (adata.obsm['spatial'][:, 0] < xlim[1]) &
        (adata.obsm['spatial'][:, 1] > ylim[0]) &
        (adata.obsm['spatial'][:, 1] < ylim[1]) &
        (adata.obs['sample'] == spsample) 
    )

    sam = adata[selection, :].copy()
    sam.obs['sample'] = sample_added
    sam.obs['barcode'] = sam.obs['barcode'].str.replace(spsample + ':', sample_added + ':')
    sam.obs['ubc'] = sam.obs['ubc'].str.replace(spsample + ':', sample_added + ':')
    sam.obs_names = sam.obs['ubc'].tolist()
    
    return sam


def rnaspc_plot_embedding_spatial(
    adata, sample, 
    sp_sample, image = 'hires', 
    channel_colors = None,
    channel_intensities = None,
    interpolation = 'nearest', # bicubic may work better when there should avoid pepper-salt like noise.
    **kwargs # to embedding plot
):
    
    if image not in adata.uns['spatial'][sp_sample]['images'].keys():
        warning(f'loaded image for this sample: {adata.uns["spatial"][sp_sample]["images"].keys()}')
        error(f'failed to find image with key `{image}`.')

    fig = rna_plot_embedding(adata[adata.obs['sample'] == sp_sample, :].copy(), sample, **kwargs)
    ax = fig.axes[0]

    # we assume that the coordinate stored in obsm['spatial'] always represent the
    # original resolution. thus, for given image key e.g. 'hires', we will first
    # query the suitable scaling factor
    xfrom, xto = ax.get_xlim()
    yfrom, yto = ax.get_ylim()
    im = adata.uns['spatial'][sp_sample]['images'][image]

    if isinstance(im, str) or isinstance(im, list):
        from exprmat.reader.spatial import get_lazyload_shape
        ymax, xmax, channel = get_lazyload_shape(im)
    else: ymax, xmax, channel = im.shape
    
    if channel_intensities is None: channel_intensities = [1] * channel
    if channel_colors is None:
        if channel == 3: channel_colors = [(1,0,0), (0,1,0), (0,0,1)] # rgb
        elif channel == 4: channel_colors = [
            (0.243, 0.459, 0.89),
            (0.839, 0.114, 0.643),
            (0.643, 0.639, 0.11),
            (0.278, 0.537, 0.071)
        ] # 10x xenium explorer default
        else: 
            warning('can not guess a channel color mapping when there is > 4 channels.')
            error('you should specify channel_colors manually.')
    
    scalefactor = adata.uns['spatial'][sp_sample]['scalefactors'][image]
    if callable(scalefactor):
        _xfrom = scalefactor(xfrom)
        _xto = scalefactor(xto)
        _yfrom = scalefactor(yfrom)
        _yto = scalefactor(yto)
    else:
        _xfrom = scalefactor * xfrom
        _xto = scalefactor * xto
        _yfrom = scalefactor * yfrom
        _yto = scalefactor * yto

    # turns to integer coordinate
    _yfrom = int(_yfrom); _yto = int(_yto)
    _xfrom = int(_xfrom); _xto = int(_xto)

    # query in image resolution
    if _xfrom < 0: _xfrom = 0
    if _xto >= xmax: _xto = xmax - 1
    if _yfrom < 0: _yfrom = 0
    if _yto >= ymax: _yto = ymax - 1
    

    # color mixer.
    mix_r = np.zeros((_yto - _yfrom, _xto - _xfrom))
    mix_g = np.zeros((_yto - _yfrom, _xto - _xfrom))
    mix_b = np.zeros((_yto - _yfrom, _xto - _xfrom))

    if isinstance(im, str) or isinstance(im, list):
        from exprmat.reader.spatial import read_fullres_from_lazyload
        im = read_fullres_from_lazyload(im, (_xfrom, _xto, _yfrom, _yto))
    else: im = im[_yfrom:_yto, _xfrom:_xto, ...]
    
    # uniform to 0-1
    if im.dtype == np.uint8:
        im = im / 255
        
    for h, c in zip(range(channel), channel_colors):
        imc = im[..., h]
        mix_r += imc * c[0] * channel_intensities[h]
        mix_g += imc * c[1] * channel_intensities[h]
        mix_b += imc * c[2] * channel_intensities[h]
    
    mix_r[mix_r > 1] = 1
    mix_g[mix_g > 1] = 1
    mix_b[mix_b > 1] = 1
    mix = np.stack([mix_r, mix_g, mix_b], axis = 2)

    ax.imshow(
        mix,
        aspect = 'auto', extent = (
            _xfrom / scalefactor, 
            _xto / scalefactor, 
            _yfrom / scalefactor, 
            _yto / scalefactor
        ), interpolation = interpolation,
        origin = 'lower'
    )

    ax.set_xlim(xfrom, xto)
    ax.set_ylim(yfrom, yto)

    return fig


def adata_filter_row_by_sum(
    adata, sample, 
    min_sum = 3, max_sum = np.finfo(np.float32).max, 
    layer = 'X'
):
    from exprmat.utils import choose_layer
    mat = choose_layer(adata, layer = layer)
    sums = mat.sum(1).reshape(-1)
    mask = (sums > min_sum) & (sums < max_sum)
    return adata[mask, :].copy()


def adata_filter_column_by_sum(
    adata, sample, 
    min_sum = 3, max_sum = np.finfo(np.float32).max, 
    layer = 'X'
):
    from exprmat.utils import choose_layer
    mat = choose_layer(adata, layer = layer)
    sums = mat.sum(0).reshape(-1)
    mask = (sums > min_sum) & (sums < max_sum)
    return adata[:, mask].copy()