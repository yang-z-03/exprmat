'''
Matcher module aims to align different versions of genomic annotations (esp. for
human and mice) into one uniform nomenclature. Names of genes frequently vary,
and less frequently do their accessions in database. Routines here is used to 
filter genes that present in common database annotations, match them to common
accessions, and align new datasets to existing ones.
'''

import os
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from exprmat.ansi import error, warning, info
from exprmat.data.finders import get_mapper_ensembl, get_mapper_name, get_genome
from exprmat.data.finders import update_mapper_ensembl, update_mapper_name, save_genome_changes
from exprmat.reader.metadata import metadata
from exprmat.configuration import default as cfg
from exprmat.ansi import info


def adjust_features(path, refine_finder = False, default_taxa = 'mmu'):
    '''
    There are three main types of features table that may occur in the matrix data folder.
    
    The currently using specification of 10x's feature files contains 3 columns for every
    feature detected, with the first column being ENSEMBL gene IDs, the second column being
    gene names, and ther third column being feature types. Feature types is an enumeration
    between ``Gene Expression``, ``Antibody Capture``, ``CRISPR Guide Capture``, 
    ``Multiplexing Capture``, or ``CUSTOM``. For multi-species data, the features and gene
    names start with the reference name, for 10x's default, e.g. ``mm10`` or ``hg19``.

    For prebuilt 10x references, the taxonomy prefixes are:

    - v1.2.0     human [hg19]     mouse [mm10]
    - v2.1.0     human [hg19]     mouse [mm10]
    - v3.0.0     human [hg19]     mouse [mm10]
    - v3.1.0     human [GRCh38]   mouse [mm10]
    - 2020a      human [GRCh38]   mouse [mm10]
    - 2024a      human [GRCh38]   mouse [GRCm39]

    For old 10x specification, when there is no antibody capture protocols been developed,
    this file may be named as ``genes.tsv``, and contains only 2 columns (before 2020.) The
    files are not compressed as they are now, and defaults to gene expression mRNA captures.

    For BGI's C4 platform, similar outputs are given in the same name as ``features.tsv.gz``,
    however, only one column indicating gene names present in this table. 

    This function helps to normalize the feature formats into the full 3-column structure,
    and compress the file if it is not done before. Since this is the only difference that
    happen to the three file components, we will enforce the format to be the newest 10x
    ones, and use the same routines to read them into memory.
    '''
    
    import pandas as pd
    fpath = os.path.join(path, 'features.tsv.gz')
    if not os.path.exists(fpath): fpath = os.path.join(path, 'features.tsv')
    if not os.path.exists(fpath): fpath = os.path.join(path, 'genes.tsv.gz')
    if not os.path.exists(fpath): fpath = os.path.join(path, 'genes.tsv')
    if not os.path.exists(fpath):
        error('do not find any possible feature table.')
        
    table = pd.read_table(fpath, sep = '\t', header = None)
    n_columns = len(table.columns)

    if n_columns >= 2 and refine_finder:

        name_finder = get_mapper_name(default_taxa)
        ensembl_finder = get_mapper_ensembl(default_taxa)

        ensembls = table[0].tolist()
        names = table[1].tolist()
        mentioned_taxa = []
        for ens, nm in zip(ensembls, names):

            # here, we will check whether the names and ensembls starts with a reference
            # prefix. 10x's default will append something like mm10_ before the gene names
            # and indices in multi-species joint analysis. if there is no such prefix detected
            # this means the gene comes from the default taxa.
            
            if '_' in ens:
                reference_name = ens.split('_')[0]
                pure_ens = ens.replace(reference_name + '_', '')
                pure_nm = nm.replace(reference_name + '_', '')
                if not reference_name in cfg['taxa.reference'].keys():
                    warning(f'gene {ens} seems to have a reference prefix, but not registered to taxa.')
                    continue

                reference_taxa = cfg['taxa.reference'][reference_name]
                alt_name_finder = get_mapper_name(reference_taxa)
                alt_ens_finder = get_mapper_ensembl(reference_taxa)
                if ((pure_nm not in alt_name_finder.keys()) and (pure_ens in alt_ens_finder.keys())):
                    update_mapper_name(reference_taxa, pure_nm, alt_ens_finder[pure_ens])
                    if reference_taxa not in mentioned_taxa: mentioned_taxa += [reference_taxa]

            elif ((nm not in name_finder.keys()) and (ens in ensembl_finder.keys())):
                update_mapper_name(default_taxa, nm, ensembl_finder[ens])
                if default_taxa not in mentioned_taxa: mentioned_taxa += [default_taxa]

        for taxa in mentioned_taxa:
            save_genome_changes(taxa)

    if n_columns == 3: return
    if n_columns == 2:

        info(f'path [{path}] contains legacy gene table.')
        construct = pd.DataFrame({
            'v0': table[0].tolist(),
            'v1': table[1].tolist(),
            'v2': ['Gene Expression'] * len(query_gname)
        })

    if n_columns == 1:

        info(f'path [{path}] contains name only gene table.')
        query_gname = table[0].tolist()
        genome = get_genome(default_taxa)
        finder_gname = genome['gene'].tolist()
        finder_ensembl = genome['ensembl'].tolist()

        query_ensembl = []
        n_not_found = 0
        for x in query_gname:

            if '_' in x:
                
                reference_name = x.split('_')[0]
                pure_nm = x.replace(reference_name + '_', '')
                
                if not reference_name in cfg['taxa.reference'].keys():
                    warning(f'gene {ens} seems to have a reference prefix, but not registered to taxa.')
                    query_ensembl += [x]
                    n_not_found += 1
                    continue

                reference_taxa = cfg['taxa.reference'][reference_name]
                alt_name_finder = get_genome(reference_taxa)['gene'].tolist()
                alt_ens_finder = get_genome(reference_taxa)['ensembl'].tolist()

                if x in finder_gname: 
                    query_ensembl += [reference_name + '_' + alt_ens_finder[alt_name_finder.index(pure_nm)]]
                    continue

            if x in finder_gname: 
                query_ensembl += [finder_ensembl[finder_gname.index(x)]]
            
            else: 
                query_ensembl += [x]
                n_not_found += 1
    
        construct = pd.DataFrame({
            'v0': query_ensembl,
            'v1': query_gname,
            'v2': ['Gene Expression'] * len(query_gname)
        })

        print(f'[{path}] {n_not_found} genes not found in reference when matching gene names.')
        print(f'[{path}] {len(finder_gname)} genes in reference genome annotation')

    construct.to_csv(
        os.path.join(path, 'features.tsv.gz'), 
        sep = '\t', header = None, index = False
    )

    fpath = os.path.join(path, 'barcodes.tsv')
    if os.path.exists(fpath):
        from sh import gzip
        gzip(fpath)
    
    fpath = os.path.join(path, 'matrix.mtx')
    if os.path.exists(fpath):
        from sh import gzip
        gzip(fpath)
    
    return
    

def read_mtx_rna(
        src: str, prefix: str, 
        metadata: metadata, sample: str, raw: bool = False,
        default_taxa = 'mmu'
    ):
    '''
    Formalize the three components in the matrix directory, and read the folder with gene
    expression data only.

    Parameters
    ----------

    src : str
        The directory to the matrix.
    
    prefix : str
        Prefix before the file name convention.
    
    metadata : exprmat.reader.metadata.metadata
        The sample specific metadata table.
    
    sample : str
        The sample name according to the metadata.

    raw : bool
        Whether this expression matrix is confirmed to be unfiltered one. Unfiltered expression
        matrix contains all detected beads or droplets, including the empty drops and doublets
        at higher probabilities. We guess this field by default to see if a sample contains
        unexpectedly high number of cells.
    '''

    # try:
    
    adjust_features(src, refine_finder = True)
    adata = sc.read_10x_mtx(
        src, var_names = 'gene_ids', gex_only = True, make_unique = False,
        prefix = prefix
    )

    # except Exception as ex: 
    #     warning('error occurs when reading matrix files:')
    #     warning(str(ex))
    #     return None

    final = match_matrix_rna(
        adata, metadata, sample, 
        force_filter = raw, default_taxa = default_taxa
    )
    
    del adata
    return final


def match_matrix_rna(
        adata, metadata: metadata, sample: str, 
        force_filter = False, default_taxa = 'mmu'
    ):

    # if more than 50000 cells in a single matrix, we just believe that it is
    # an unfiltered raw matrix. we should roughly filter the empty droplets

    if adata.n_obs > 50000 or force_filter:
        valid = (adata.X.sum(axis = 1).transpose() > 200).tolist()[0]
        adata_f = adata[valid, :]
    else: adata_f = adata

    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    rows = rows[rows['modality'] == 'rna']
    assert len(rows) == 1
    props = rows.iloc[0]

    # append the sample name to the barcode
    adata_f.obs_names = props['sample'] + ':' + adata_f.obs_names

    # map gene naming
    gname = adata_f.var_names.tolist()
    names = []
    gmask = []

    # here, we just add another condition to test whether the gname list is appropriate
    # ensembl format. if it is not, we try to map genes directly onto the names.
    # though i specify the var_names should be 'gene_ids', it may occur exceptions
    # where there are man-made references containing two or more species. by convention
    # in these double species reference, the 'gene_ids' should be 'mm10_ENSMUSG...'
    # or just name of the genes.

    default_finder_ens = get_mapper_ensembl(default_taxa)
    default_finder_name = get_mapper_name(default_taxa)
    not_in_list = []

    for x in gname:

        if '_' in x:

            reference_name = x.split('_')[0]
            pure_nm = x.replace(reference_name + '_', '')

            if not reference_name in cfg['taxa.reference'].keys():
                warning(f'gene {x} seems to have a reference prefix, but not registered to taxa.')
                gmask.append(False)
                not_in_list.append(x)
                continue

            reference_taxa = cfg['taxa.reference'][reference_name]
            alt_finder_name = get_mapper_name(reference_taxa)
            alt_finder_ens = get_mapper_ensembl(reference_taxa)

            if pure_nm in alt_finder_ens.keys():
                gmask.append(True)
                names.append(alt_finder_ens[pure_nm])
                continue
            
            if pure_nm in alt_finder_name.keys():
                gmask.append(True)
                names.append(alt_finder_name[pure_nm])
                continue
        
        if x in default_finder_ens.keys():
            gmask.append(True)
            names.append(default_finder_ens[x])
            continue
        
        if x in default_finder_name.keys():
            gmask.append(True)
            names.append(default_finder_name[x])
            continue
            
        gmask.append(False)
        not_in_list.append(x)

    final = adata_f[:, gmask].copy()
    final.var_names = ['rna:' + x for x in names]
    # remove duplicated genes
    duplicated = set(final.var_names[final.var_names.duplicated()].tolist())
    final = final[:, final.var_names.duplicated() == False].copy()

    info(f'{len(not_in_list)} genes (out of {len(gname)}) not in the reference gene list.')
    info(f'total {len(names)} genes mapped. {len(np.unique(names))} unique genes.')

    # attach cell metadata onto the obs slot.
    
    for k in props.index.tolist():
        if k not in ['location']: final.obs[k] = props[k]
    
    final.obs['barcode'] = final.obs_names
    final.obs['ubc'] = [props['sample'] + ':' + str(x + 1) for x in range(final.n_obs)]
    final.obs_names = [props['sample'] + ':' + str(x + 1) for x in range(final.n_obs)]
    del final.var
    
    if not isinstance(final.X, csr_matrix):
        final.X = csr_matrix(final.X)

    del adata_f
    return final
