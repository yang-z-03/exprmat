
from exprmat.grn.infergrn import adjacencies

# pruning targets
import pandas as pd
from exprmat.ansi import error, info
from exprmat.utils import choose_layer
from exprmat.grn.prune import prune as prune_to_df
from exprmat.grn.modules import modules_from_adjacencies
from exprmat.data.tf import get_ranking_dbs, get_motif_annotation_fname


def prune(
    adata,
    adjacency_table,
    taxa,
    layer: str = 'counts', gene = 'gene',
    features: str = 'genes',
    cistromes: str = 'motifs',
    thresholds = [0.75, 0.90],
    top_n_targets = [50],
    top_n_regulators = [5, 10, 50],
    min_genes = 20,
    mask_dropout = True,
    keep_only_activating = True,
    rank_threshold = 5000,
    auc_threshold = 0.05,
    nes_threshold = 3.0,
    max_similarity_fdr = 0.001,
    min_orthologous_identity = 0,
    chunk_size = 100,
    ncpus = 1,
    verbose = False
):
    ex_mtx = choose_layer(adata, layer = layer)
    ex_mtx = pd.DataFrame(ex_mtx.todense())
    ex_mtx.columns = adata.var[gene].tolist()

    coexp_modules = modules_from_adjacencies(
        adjacency_table,
        ex_mtx,
        thresholds = thresholds,
        top_n_targets = top_n_targets, 
        top_n_regulators = top_n_regulators,
        min_genes = min_genes,
        rho_mask_dropouts = mask_dropout,
        keep_only_activating = keep_only_activating,
    )

    if len(coexp_modules) == 0:
        error('not a single coexpression module detected.')

    dbs = get_ranking_dbs(taxa, features = features, cistromes = cistromes)
    motif_annotation_fname = get_motif_annotation_fname(taxa)
    
    df_motifs = prune_to_df(
        dbs,
        coexp_modules,
        motif_annotation_fname,
        rank_threshold = rank_threshold,
        auc_threshold = auc_threshold,
        nes_threshold = nes_threshold,
        module_chunksize = chunk_size,
        num_workers = ncpus,
        motif_similarity_fdr = max_similarity_fdr,
        orthologuous_identity_threshold = min_orthologous_identity,
        verbose = verbose
    )

    return df_motifs


# auc calculations
from exprmat.descriptive.aucell import aucell
from exprmat.grn.utils import df_to_regulons

def auc_signature(
    adata,
    signatures,
    layer: str = 'counts', gene = 'gene',
    auc_threshold = 0.05,
    weighted = False,
    ncpus = 1,
    seed = 42
):
    ex_mtx = choose_layer(adata, layer = layer)
    ex_mtx = pd.DataFrame(ex_mtx.todense())
    ex_mtx.columns = adata.var[gene].tolist()

    auc_mtx = aucell(
        ex_mtx,
        signatures,
        auc_threshold = auc_threshold,
        noweights = not weighted,
        seed = seed,
        num_workers = ncpus,
    )

    auc_mtx.index = adata.obs_names.tolist()
    return auc_mtx