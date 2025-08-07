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
from exprmat.data.finders import update_mapper_ensembl, update_mapper, save_genome_changes
from exprmat.reader.metadata import metadata
from exprmat.configuration import default as cfg
from exprmat.ansi import info


def refine_finder(features_fpath, taxa = 'mmu'):
    table = pd.read_table(features_fpath, sep = '\t', header = None)
    assert len(table.columns) >= 2

    name_finder = get_mapper_name(taxa)
    ensembl_finder = get_mapper_ensembl(taxa)

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
            if not reference_name.lower() in cfg['taxa.reference'].keys():
                warning(f'gene {ens} seems to have a reference prefix, but not registered to taxa.')
                continue

            reference_taxa = cfg['taxa.reference'][reference_name.lower()]
            alt_name_finder = get_mapper_name(reference_taxa)
            alt_ens_finder = get_mapper_ensembl(reference_taxa)
            if ((pure_nm not in alt_name_finder.keys()) or (pure_ens not in alt_ens_finder.keys())):
                update_mapper(reference_taxa, pure_nm, pure_ens)
                if reference_taxa not in mentioned_taxa: mentioned_taxa += [reference_taxa]

        elif ((nm not in name_finder.keys()) or (ens not in ensembl_finder.keys())):
            update_mapper(taxa, nm, ens)
            if taxa not in mentioned_taxa: mentioned_taxa += [taxa]

    for taxa in mentioned_taxa:
        save_genome_changes(taxa)


def adjust_features(path, refine_finder = False, default_taxa = 'mmu', eccentric = None):
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
            
            if eccentric is not None: ens = eccentric(ens)
            if '_' in ens:
                reference_name = ens.split('_')[0]
                pure_ens = ens.replace(reference_name + '_', '')
                pure_nm = nm.replace(reference_name + '_', '')
                if not reference_name.lower() in cfg['taxa.reference'].keys():
                    warning(f'gene {ens} seems to have a reference prefix, but not registered to taxa.')
                    continue

                reference_taxa = cfg['taxa.reference'][reference_name.lower()]
                alt_name_finder = get_mapper_name(reference_taxa)
                alt_ens_finder = get_mapper_ensembl(reference_taxa)
                if ((pure_nm not in alt_name_finder.keys()) or (pure_ens not in alt_ens_finder.keys())):
                    update_mapper(reference_taxa, pure_nm, pure_ens)
                    if reference_taxa not in mentioned_taxa: mentioned_taxa += [reference_taxa]

            elif ((nm not in name_finder.keys()) or (ens not in ensembl_finder.keys())):
                update_mapper(default_taxa, nm, ens)
                if default_taxa not in mentioned_taxa: mentioned_taxa += [default_taxa]

        for taxa in mentioned_taxa:
            save_genome_changes(taxa)

    if n_columns >= 3: 

        fpath = os.path.join(path, 'genes.tsv')
        if os.path.exists(fpath):
            os.rename(os.path.join(path, 'genes.tsv'), os.path.join(path, 'features.tsv'))
        
        fpath = os.path.join(path, 'features.tsv')
        if os.path.exists(fpath):
            from sh import gzip
            gzip(fpath)

        fpath = os.path.join(path, 'barcodes.tsv')
        if os.path.exists(fpath):
            from sh import gzip
            gzip(fpath)

        fpath = os.path.join(path, 'matrix.mtx')
        if os.path.exists(fpath):
            from sh import gzip
            gzip(fpath)

        return
    

    if n_columns == 2:

        info(f'path [{path}] contains legacy gene table.')
        construct = pd.DataFrame({
            'v0': table[0].tolist(),
            'v1': table[1].tolist(),
            'v2': ['Gene Expression'] * len(table[1].tolist())
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
            
            if eccentric is not None: x = eccentric(x)
            if '_' in x:
                
                reference_name = x.split('_')[0]
                pure_nm = x.replace(reference_name + '_', '')
                
                if not reference_name.lower() in cfg['taxa.reference'].keys():
                    warning(f'gene {x} seems to have a reference prefix, but not registered to taxa.')
                    query_ensembl += [x]
                    n_not_found += 1
                    continue

                reference_taxa = cfg['taxa.reference'][reference_name.lower()]
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

        info(f'[adjust] {n_not_found} genes not found in reference when matching gene names.')
        info(f'[adjust] {len(finder_gname)} genes in reference genome annotation')

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
    
    fpath = os.path.join(path, 'genes.tsv')
    if os.path.exists(fpath):
        os.rename(fpath, os.path.join(path, 'genes.tsv.backup'))

    return
    

def read_mtx_rna(
    src: str, prefix: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None, suppress_filter = False,
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
    
    adjust_features(src, refine_finder = True, default_taxa = default_taxa, eccentric = eccentric)

    import warnings
    warnings.filterwarnings('ignore')

    adata = sc.read_10x_mtx(
        src, var_names = 'gene_ids', gex_only = True, make_unique = False,
        prefix = prefix
    )

    if eccentric is not None:
        adata.var_names = [eccentric(x) for x in adata.var_names]

    # except Exception as ex: 
    #     warning('error occurs when reading matrix files:')
    #     warning(str(ex))
    #     return None

    final = match_matrix_rna(
        adata, metadata, sample, suppress_filter = suppress_filter,
        force_filter = raw, default_taxa = default_taxa
    )

    warnings.filterwarnings('default')
    
    del adata
    return final


def read_h5_rna(
    path: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None, suppress_filter = False,
):
    import warnings
    warnings.filterwarnings('ignore')

    adata = sc.read_10x_h5(
        path, gex_only = True
    )

    adata.var_names = adata.var['gene_ids'].tolist()
    del adata.var

    if eccentric is not None:
        adata.var_names = [eccentric(x) for x in adata.var_names]

    # except Exception as ex: 
    #     warning('error occurs when reading matrix files:')
    #     warning(str(ex))
    #     return None

    final = match_matrix_rna(
        adata, metadata, sample, suppress_filter = suppress_filter,
        force_filter = raw, default_taxa = default_taxa
    )

    warnings.filterwarnings('default')
    
    del adata
    return final


def read_table_rna(
    src: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None
):
    '''
    Read from compressed or plain text tables.

    Parameters
    ----------

    src : str
        The directory to the matrix.
    
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
    
    # in csv/tsv texts, genes are rows and cells are columns.
    adata = sc.read_text(
        src, delimiter = '\t', first_column_names = True, dtype = 'float32'
    ).T

    if eccentric is not None:
        adata.var_names = [eccentric(x) for x in adata.var_names]

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


def read_h5ad_rna(
        src: str, 
        metadata: metadata, sample: str, raw: bool = False,
        default_taxa = 'mmu'
    ):
    '''
    Read from a given h5ad. This typically occurs when you have to prepare the data yourself
    from others who give you a complex atlas. The assignment of samples and batches happens
    within a single h5ad file. Then, you would like to keep nearly all of the metadata you
    prepared by hand as it is. Only `modality`, `taxa` and `ubc` are generated for you.

    Thus, you must manually supply `sample`, `batch` and `group` yourself. Otherwise the program
    may not work as usual. It permits multiple samples in one h5ad file.

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
    
    adata = sc.read_h5ad(src)

    # except Exception as ex: 
    #     warning('error occurs when reading matrix files:')
    #     warning(str(ex))
    #     return None

    if adata.var_names.tolist()[0].startswith('rna:'):
        info('it seems that you have prepared the h5ad file manually.')
        info('we will not alter anything and test mandatory columns as is.')
        assert 'sample' in adata.obs.keys()
        assert 'modality' in adata.obs.keys()
        assert 'taxa' in adata.obs.keys()
        assert 'batch' in adata.obs.keys()
        assert 'gene' in adata.var.keys()
        return adata
    
    final = match_matrix_rna(
        adata, metadata, sample, 
        force_filter = raw, default_taxa = default_taxa,
        # this switch tells the program not to alter the cell names (keep them
        # directly to `barcodes`, not adding prefix to them), and keep the
        # sample, batch, and group columns as is. 
        do_not_alter_obs_names = True
    )
    
    del adata
    return final


def convert_to_ugene(gname, taxa):
    
    # map gene naming
    names = []
    gmask = []

    # here, we just add another condition to test whether the gname list is appropriate
    # ensembl format. if it is not, we try to map genes directly onto the names.
    # though i specify the var_names should be 'gene_ids', it may occur exceptions
    # where there are man-made references containing two or more species. by convention
    # in these double species reference, the 'gene_ids' should be 'mm10_ENSMUSG...'
    # or just name of the genes.

    default_finder_ens = get_mapper_ensembl(taxa)
    default_finder_name = get_mapper_name(taxa)
    not_in_list = []

    for x in gname:

        if '_' in x:

            reference_name = x.split('_')[0]
            pure_nm = x.replace(reference_name + '_', '')

            if not reference_name.lower() in cfg['taxa.reference'].keys():
                warning(f'gene {x} seems to have a reference prefix, but not registered to taxa.')
                gmask.append(False)
                not_in_list.append(x)
                continue

            reference_taxa = cfg['taxa.reference'][reference_name.lower()]
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

    names = ['rna:' + x for x in names]
    return names, gmask, not_in_list


def match_matrix_rna(
    adata, metadata: metadata, sample: str, 
    force_filter = False, default_taxa = 'mmu',
    do_not_alter_obs_names = False, suppress_filter = False
):

    # if more than 50000 cells in a single matrix, we just believe that it is
    # an unfiltered raw matrix. we should roughly filter the empty droplets

    if (adata.n_obs > 50000 or force_filter) and (not suppress_filter):
        valid = (adata.X.sum(axis = 1).transpose() > 200).tolist()[0]
        adata_f = adata[valid, :]

    else: adata_f = adata

    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    rows = rows[
        (rows['modality'] == 'rna') |
        (rows['modality'] == 'rnasp-c') |
        (rows['modality'] == 'rnasp-b') |
        (rows['modality'] == 'rnasp-s')
    ]
    assert len(rows) == 1
    props = rows.iloc[0]

    if not do_not_alter_obs_names:
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

            if not reference_name.lower() in cfg['taxa.reference'].keys():
                warning(f'gene {x} seems to have a reference prefix, but not registered to taxa.')
                gmask.append(False)
                not_in_list.append(x)
                continue

            reference_taxa = cfg['taxa.reference'][reference_name.lower()]
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
        if do_not_alter_obs_names:
            if k in ['sample', 'batch', 'group']: continue
        if k not in ['location']: final.obs[k] = props[k]
    
    final.obs['barcode'] = final.obs_names
    final.obs['ubc'] = [props['sample'] + ':' + str(x + 1) for x in range(final.n_obs)]
    final.obs_names = [props['sample'] + ':' + str(x + 1) for x in range(final.n_obs)]
    del final.var
    
    if not isinstance(final.X, csr_matrix):
        final.X = csr_matrix(final.X)

    del adata_f
    return final


def stringify_tcr(df, contig, v, d, j, nt, aa):
    clone = ""
    df = df.copy()
    df.fillna('na', inplace = True)
    if v is not None and j is not None:
        if d is not None: clone = 'vdj(' + df[v] + ', ' + df[d] + ', ' + df[j] + ')'
        else: clone = 'vj(' + df[v] + ', ' + df[j] + ')'
    
    # aa alone may show degrees of degeneration!
    if nt is not None: clone = clone + (', ' if clone != '' else '') + 'nt(' + df[nt] + ')'
    elif aa is not None: clone = clone + (', ' if clone != '' else '') + 'aa(' + df[aa] + ')'
    return f'{contig}(' + clone + ')' 


def parse_tcr_percell(
    fpath, *, sep = ',', index = True,

    # the mapper of columns between some custom export files towards a uniform one.
    barcode = 'barcode',
    sample = 'sample',
    barcode_translation = None,

    trav_gene = None,
    trad_gene = None,
    traj_gene = None,
    tra_full_length = None,
    tra_productive = None,
    tra_umi = None,
    tra_reads = None,
    tra_nt = None,
    tra_aa = None,

    trbv_gene = None,
    trbd_gene = None,
    trbj_gene = None,
    trb_full_length = None,
    trb_productive = None,
    trb_umi = None,
    trb_reads = None,
    trb_nt = None,
    trb_aa = None
):
    if isinstance(fpath, str):
        raw_table = pd.read_table(fpath, index_col = 0 if index else None, sep = sep)
    else: raw_table = fpath

    barcodes = raw_table[barcode].tolist()
    if barcode_translation is not None:
        barcodes = [barcode_translation(x) for x in barcodes]
    
    tidy = { 'barcode': barcodes, 'sample': raw_table[sample].tolist() }
    tidy = pd.DataFrame(tidy)

    if trav_gene is not None: tidy['trav'] = raw_table[trav_gene].tolist()
    if trad_gene is not None: tidy['trad'] = raw_table[trad_gene].tolist()
    if traj_gene is not None: tidy['traj'] = raw_table[traj_gene].tolist()
    if tra_full_length is not None: tidy['tra.full'] = raw_table[tra_full_length].tolist()
    if tra_productive is not None: tidy['tra.productive'] = raw_table[tra_productive].tolist()
    if tra_umi is not None: tidy['tra.umi'] = raw_table[tra_umi].tolist()
    if tra_reads is not None: tidy['tra.reads'] = raw_table[tra_reads].tolist()
    if tra_nt is not None: tidy['tra.nt'] = raw_table[tra_nt].tolist()
    if tra_aa is not None: tidy['tra.aa'] = raw_table[tra_aa].tolist()

    if trbv_gene is not None: tidy['trbv'] = raw_table[trbv_gene].tolist()
    if trbd_gene is not None: tidy['trbd'] = raw_table[trbd_gene].tolist()
    if trbj_gene is not None: tidy['trbj'] = raw_table[trbj_gene].tolist()
    if trb_full_length is not None: tidy['trb.full'] = raw_table[trb_full_length].tolist()
    if trb_productive is not None: tidy['trb.productive'] = raw_table[trb_productive].tolist()
    if trb_umi is not None: tidy['trb.umi'] = raw_table[trb_umi].tolist()
    if trb_reads is not None: tidy['trb.reads'] = raw_table[trb_reads].tolist()
    if trb_nt is not None: tidy['trb.nt'] = raw_table[trb_nt].tolist()
    if trb_aa is not None: tidy['trb.aa'] = raw_table[trb_aa].tolist()

    present_a = \
        (trav_gene is not None) or \
        (trad_gene is not None) or \
        (traj_gene is not None) or \
        (tra_nt is not None) or \
        (tra_aa is not None)
    
    present_b = \
        (trbv_gene is not None) or \
        (trbd_gene is not None) or \
        (trbj_gene is not None) or \
        (trb_nt is not None) or \
        (trb_aa is not None)
    
    if present_a: tidy['tra'] = stringify_tcr(
        tidy, 'tra', 
        'trav'   if trav_gene is not None else None, 
        'trad'   if trad_gene is not None else None, 
        'traj'   if traj_gene is not None else None, 
        'tra.nt' if tra_nt is not None else None, 
        'tra.aa' if tra_aa is not None else None
    )
        
    if present_b: tidy['trb'] = stringify_tcr(
        tidy, 'trb', 
        'trbv'   if trbv_gene is not None else None, 
        'trbd'   if trbd_gene is not None else None, 
        'trbj'   if trbj_gene is not None else None, 
        'trb.nt' if trb_nt is not None else None, 
        'trb.aa' if trb_aa is not None else None
    )
    
    if present_a and present_b:
        tidy['clone'] = 'tcr(' + tidy['tra'] + ', ' + tidy['trb'] + ')'
    elif present_a: tidy['clone'] = tidy['tra']
    elif present_b: tidy['clone'] = tidy['trb']

    # assign a clone id for each.
    indices = []
    existing = {}
    next_id = 1
    clones = tidy['clone'].tolist()
    for c in clones:
        if c not in existing.keys():
            indices.append('c:' + str(next_id))
            existing[c] = 'c:' + str(next_id)
            next_id += 1
        else: indices.append(existing[c])
    
    tidy['clone.id'] = indices
    return tidy


def parse_tcr_10x(
    fpath, sep = ',', barcode_translation = None, sample = '.',
    filter_non_productive = True,
    filter_non_full_length = True,
):

    if isinstance(fpath, str):
        raw_table = pd.read_table(fpath, index_col = None, sep = sep)
    else: raw_table = fpath

    # filter out illegal detections
    raw_table = raw_table.loc[raw_table['is_cell'] == True, :]
    raw_table = raw_table.loc[raw_table['high_confidence'] == True, :]
    if filter_non_full_length: 
        raw_table = raw_table.loc[raw_table['full_length'] == True, :]
    if filter_non_productive:
        raw_table = raw_table.loc[raw_table['productive'] == True, :]

    tra_contigs = raw_table.loc[
        raw_table['chain'] == 'TRA', [
            'barcode', 'v_gene', 'd_gene', 'j_gene', 'full_length', 'productive', 
            'reads', 'umis', 'cdr3_nt', 'cdr3'
        ]
    ].copy()

    trb_contigs = raw_table.loc[
        raw_table['chain'] == 'TRB', [
            'barcode', 'v_gene', 'd_gene', 'j_gene', 'full_length', 'productive', 
            'reads', 'umis', 'cdr3_nt', 'cdr3'
        ]
    ].copy()

    tra_contigs = tra_contigs.rename(columns = {
        'v_gene'      : 'trav', 
        'd_gene'      : 'trad', 
        'j_gene'      : 'traj', 
        'full_length' : 'tra.full', 
        'productive'  : 'tra.productive', 
        'reads'       : 'tra.reads',
        'umis'        : 'tra.umi',
        'cdr3_nt'     : 'tra.nt', 
        'cdr3'        : 'tra.aa'
    })

    trb_contigs = trb_contigs.rename(columns = {
        'v_gene'      : 'trbv', 
        'd_gene'      : 'trbd', 
        'j_gene'      : 'trbj', 
        'full_length' : 'trb.full', 
        'productive'  : 'trb.productive', 
        'reads'       : 'trb.reads',
        'umis'        : 'trb.umi',
        'cdr3_nt'     : 'trb.nt', 
        'cdr3'        : 'trb.aa'
    })

    tra_contigs.index = tra_contigs['barcode'].tolist()
    del tra_contigs['barcode']
    trb_contigs.index = trb_contigs['barcode'].tolist()
    del trb_contigs['barcode']

    # set to inner, should detect both alpha and beta chains
    tidy = tra_contigs.join(trb_contigs, how = 'inner')
    tidy['tra'] = stringify_tcr(tidy, 'tra', 'trav', 'trad', 'traj', 'tra.nt', 'tra.aa')
    tidy['trb'] = stringify_tcr(tidy, 'trb', 'trbv', 'trbd', 'trbj', 'trb.nt', 'trb.aa')
    tidy['clone'] = 'tcr(' + tidy['tra'] + ', ' + tidy['trb'] + ')'

    # assign a clone id for each.
    indices = []
    existing = {}
    next_id = 1
    clones = tidy['clone'].tolist()
    for c in clones:
        if c not in existing.keys():
            indices.append('c:' + str(next_id))
            existing[c] = 'c:' + str(next_id)
            next_id += 1
        else: indices.append(existing[c])
    
    tidy['clone.id'] = indices

    barcodes = tidy.index.tolist()
    if barcode_translation is not None:
        barcodes = [barcode_translation(x) for x in barcodes]
    tidy['barcode'] = barcodes
    tidy['sample'] = sample
    
    return tidy


def attach_tcr(adata, fpath):

    info(f'reading tcr table from {fpath} ...')
    tcr = pd.read_table(fpath, index_col = None, sep = '\t')
    assert 'barcode' in tcr.columns
    assert 'sample' in tcr.columns
    assert 'tra' in tcr.columns
    assert 'trb' in tcr.columns
    origin_name = adata.obs_names.tolist()

    obs_df = adata.obs.copy()
    # obs_df.index = (adata.obs['sample'].astype('str') + ':' + adata.obs['barcode'].astype('str')).tolist()
    obs_df.index = adata.obs['barcode'].astype('str').tolist()

    tra = adata.obs['tra'].tolist() if 'tra' in adata.obs.keys() else ['na'] * adata.n_obs
    trb = adata.obs['trb'].tolist() if 'trb' in adata.obs.keys() else ['na'] * adata.n_obs
    clone = adata.obs['clone'].tolist() if 'clone' in adata.obs.keys() else ['na'] * adata.n_obs
    cloneid = adata.obs['clone.id'].tolist() if 'clone.id' in adata.obs.keys() else ['na'] * adata.n_obs
    trav = adata.obs['trav'].tolist() if 'trav' in adata.obs.keys() else ['na'] * adata.n_obs
    traj = adata.obs['traj'].tolist() if 'traj' in adata.obs.keys() else ['na'] * adata.n_obs
    trbv = adata.obs['trbv'].tolist() if 'trbv' in adata.obs.keys() else ['na'] * adata.n_obs
    trbj = adata.obs['trbj'].tolist() if 'trbj' in adata.obs.keys() else ['na'] * adata.n_obs
    
    # clean the rows out
    if 'tra' in obs_df.keys(): del obs_df['tra']
    if 'trb' in obs_df.keys(): del obs_df['trb']
    if 'trav' in obs_df.keys(): del obs_df['trav']
    if 'traj' in obs_df.keys(): del obs_df['traj']
    if 'trbv' in obs_df.keys(): del obs_df['trbv']
    if 'trbj' in obs_df.keys(): del obs_df['trbj']
    if 'clone' in obs_df.keys(): del obs_df['clone']
    if 'clone.id' in obs_df.keys(): del obs_df['clone.id']
    
    tcr.index = tcr['sample'].astype('str') + ':' + tcr['barcode'].astype('str')
    # test if tcr contains duplicated chains:
    dup_count = tcr.index.duplicated().sum()
    if dup_count > 0: 
        warning(f'tcr table contains {dup_count} duplicated contigs for one cell.')
        sort_tcr = tcr.sort_values(by = ['tra.umi', 'trb.umi'], ascending = False)
        tcr = sort_tcr.loc[~ sort_tcr.index.duplicated(), :]

    selected_cols = ['tra', 'trb', 'clone', 'clone.id']
    if 'trav' in tcr.columns: selected_cols += ['trav']
    if 'traj' in tcr.columns: selected_cols += ['traj']
    if 'trbv' in tcr.columns: selected_cols += ['trbv']
    if 'trbj' in tcr.columns: selected_cols += ['trbj']

    new_df = obs_df.join(tcr.loc[:, selected_cols], how = 'left')
    n_match = len(new_df) - new_df['clone'].isna().sum()
    new_df.index = origin_name

    # make sure the join operation do not introduce duplicates.
    assert len(new_df) == len(origin_name)

    n_tra = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['tra'].tolist(), tra)] 
    n_trb = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['trb'].tolist(), trb)] 
    n_clone = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['clone'].tolist(), clone)] 
    n_cloneid = [y if str(x) == 'nan' else (s + ':' + str(x)) for x, s, y in zip(new_df['clone.id'].tolist(), new_df['sample'].tolist(), cloneid)] 
    n_trav = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['trav'].tolist(), trav)] 
    n_traj = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['traj'].tolist(), traj)] 
    n_trbv = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['trbv'].tolist(), trbv)] 
    n_trbj = [y if str(x) == 'nan' else str(x) for x, y in zip(new_df['trbj'].tolist(), trbj)] 

    new_df['tra'] = n_tra
    new_df['trb'] = n_trb
    new_df['clone'] = n_clone
    new_df['clone.id'] = n_cloneid
    new_df['trav'] = n_trav
    new_df['traj'] = n_traj
    new_df['trbv'] = n_trbv
    new_df['trbj'] = n_trbj

    adata.obs = new_df

    warning(f'{n_match} out of {len(tcr)} ({(100 * n_match / len(tcr)):.1f}%) tcr detections mapped.')
    return


def attach_splice_reads_mtx(adata, folder, default_taxa, sample):
    '''
    Attach MGI DNBC4 flavor output spliced/unspliced matrices. The folder of spliced reads contains
    barcodes.tsv.gz, features.tsv.gz, spanning, spliced and unspliced matrix files. Note that the
    order and content of the features may not match the given adata.
    '''

    features = pd.read_table(os.path.join(folder, 'features.tsv.gz'), sep = '\t', header = None)
    features = features.iloc[:, 0].tolist()

    from scipy.io import mmread
    from scipy.sparse import csr_matrix
    import gzip
    import io

    with gzip.open(os.path.join(folder, 'spliced.mtx.gz'), 'rb') as f:
        with io.TextIOWrapper(f, encoding = 'utf-8') as enc:
            spliced = mmread(enc)
        
    with gzip.open(os.path.join(folder, 'unspliced.mtx.gz'), 'rb') as f:
        with io.TextIOWrapper(f, encoding = 'utf-8') as enc:
            unspliced = mmread(enc)
    
    with gzip.open(os.path.join(folder, 'spanning.mtx.gz'), 'rb') as f:
        with io.TextIOWrapper(f, encoding = 'utf-8') as enc:
            ambiguous = mmread(enc)
    
    barcodes = pd.read_table(os.path.join(folder, 'barcodes.tsv.gz'), sep = '\t', header = None)
    barcodes = barcodes.iloc[:, 0].tolist()

    # make annotated data
    adata_f = ad.AnnData(
        X = csr_matrix((len(barcodes), len(features)), dtype = np.float32)
    )
    
    adata_f.obs_names = barcodes
    adata_f.var_names = features
    adata_f.layers['spliced'] = spliced.T.tocsr()
    adata_f.layers['unspliced'] = unspliced.T.tocsr()
    adata_f.layers['ambiguous'] = ambiguous.T.tocsr()

    # map gene naming
    gname = features
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

            if not reference_name.lower() in cfg['taxa.reference'].keys():
                warning(f'gene {x} seems to have a reference prefix, but not registered to taxa.')
                gmask.append(False)
                not_in_list.append(x)
                continue

            reference_taxa = cfg['taxa.reference'][reference_name.lower()]
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
    del adata_f
    final.var_names = ['rna:' + x for x in names]
    # remove duplicated genes
    duplicated = set(final.var_names[final.var_names.duplicated()].tolist())
    final = final[:, final.var_names.duplicated() == False].copy()
    final.obs_names = sample + ':' + final.obs_names

    # by now the variable names are identical, while the obs names should match
    # those in the obs['barcode']

    adata = adata[
        [x in final.obs_names for x in adata.obs['barcode'].tolist()],
        [x in final.var_names for x in adata.var_names.tolist()]
    ].copy()

    final = final[adata.obs['barcode'], adata.var_names].copy()
    adata.layers['spliced'] = final.layers['spliced']
    adata.layers['unspliced'] = final.layers['unspliced']
    adata.layers['ambiguous'] = final.layers['ambiguous']

    return adata


def attach_splice_reads_loom(adata, loom_file, default_taxa, sample):
    '''
    Attach Loom format of spliced/unspliced matrices.
    Typically from outputs of velocyto.
    '''

    # this typically gives a following result:
    # AnnData object with n_obs × n_vars = 55321 × 33696
    #   var: 'Accession', 'Chromosome', 'End', 'Start', 'Strand'
    #   layers: 'ambiguous', 'matrix', 'spliced', 'unspliced'

    adata_f = sc.read_loom(
        loom_file, sparse = True, cleanup = False, 
        X_name = 'matrix', obs_names = 'CellID', obsm_names = None, 
        var_names = 'Gene', varm_names = None, dtype = 'float32'
    )

    features = adata_f.var_names.tolist()
    # rename observations
    observs = [x[x.index(':') + 1:-1] for x in adata_f.obs_names]
    adata_f.obs_names = observs

    # map gene naming
    gname = features
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

            if not reference_name.lower() in cfg['taxa.reference'].keys():
                warning(f'gene {x} seems to have a reference prefix, but not registered to taxa.')
                gmask.append(False)
                not_in_list.append(x)
                continue

            reference_taxa = cfg['taxa.reference'][reference_name.lower()]
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
    del adata_f
    final.var_names = ['rna:' + x for x in names]
    # remove duplicated genes
    duplicated = set(final.var_names[final.var_names.duplicated()].tolist())
    final = final[:, final.var_names.duplicated() == False].copy()
    final.obs_names = sample + ':' + final.obs_names

    # by now the variable names are identical, while the obs names should match
    # those in the obs['barcode']

    # remove 10x lane no.
    adata.obs['barcode.pure'] = [
        x[:-2] if x[-2] == '-' else x for x in adata.obs['barcode'].tolist()
    ]

    adata = adata[
        [x in final.obs_names for x in adata.obs['barcode.pure'].tolist()],
        [x in final.var_names for x in adata.var_names.tolist()]
    ].copy()

    final = final[adata.obs['barcode.pure'], adata.var_names].copy()
    adata.layers['spliced'] = final.layers['spliced']
    adata.layers['unspliced'] = final.layers['unspliced']
    adata.layers['ambiguous'] = final.layers['ambiguous']
    del adata.obs['barcode.pure']
    return adata