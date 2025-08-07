'''
Experiments may be carried out with different designs. There are two main types of experimental 
designs to consider in terms of transcriptomic studies (or other assays that aim to measure
sectional cellular states): (1) the one involves timing, for time sequence studies and lineage
tracing studies, (2) and the one that do not involve timing, merely focusing on differences in 
different experimental conditions, genetic background etc. Interestingly, lineage tracing-related
studies can fail to capture both output of cell states, this may intervene the experiment at a 
previous timepoint, and gather tracers at later timepoints, yielding only one time point observation.
This type of study should be time-related study, but with only one known timepoint, leaving the
history to be inferred only.

Experiment finds and organize the data from a given metadata table, distinguishing between different
samples, batches, modalities, and time series, and normalize them accordingly. Same sample from
different modalities will be merged into a mudata here, but different samples are kept separately
for sample-level QC is not performed yet.
'''

import scanpy as sc
import anndata as ad
import mudata as mu
import pandas as pd
import numpy as np
import os

from exprmat.reader.metadata import metadata, load_metadata
from exprmat.data.finders import get_genome
from exprmat.reader.matcher import (
    read_mtx_rna, read_h5ad_rna, read_table_rna, 
    parse_tcr_10x, attach_splice_reads_mtx, attach_splice_reads_loom
)
from exprmat.reader.matcher import attach_tcr
from exprmat.ansi import warning, info, error, red, green
from exprmat.configuration import default as cfg
import exprmat.reader.static as st


class experiment:
    
    def __init__(
        self, meta : metadata, 
        eccentric = None, 
        mudata = None, modalities = {}, 
        dump = '.', subset = None
    ):

        # TODO: we support rna only at present.
        table = meta.dataframe.to_dict(orient = 'list')
        self.mudata = mudata
        self.modalities = modalities
        self.metadata = meta
        self.subset = subset
        self.directory = dump

        if self.mudata is not None:
            if len(self.modalities) == 0:
                warning('samples are not dumped in the experiment directory.')
            return
        
        if len(self.modalities) > 0:
            if self.mudata is None:
                warning('integrated mudata object is not generated.')
            return

        self.modalities = {}
        for i_loc, i_sample, i_batch, i_grp, i_mod, i_taxa in zip(
            table['location'], table['sample'], table['batch'], table['group'],
            table['modality'], table['taxa']
        ):
            
            # extension of taxa: you can specify a specific version of the genome
            # assembly when specifying taxa. leaving it out allows exprmat to choose
            # the assembly (the latest version) automatically.

            # e.g.
            #    mmu            = 'grcm39'
            #    mmu/grcm38     = 'grcm38'

            if '/' in i_taxa: i_taxa, i_assembly = i_taxa.split('/')
            else: i_assembly = cfg['default.assembly'][i_taxa]
            
            info(f'reading sample {i_sample} [{i_mod}] ...')

            attempt_path = os.path.join(dump, i_mod, i_sample + '.h5ad')
            if os.path.exists(attempt_path):
                if not i_mod in self.modalities.keys(): self.modalities[i_mod] = {}
                self.modalities[i_mod][i_sample] = sc.read_h5ad(attempt_path)
                warning(f'load pre-exisiting file `{i_mod}/{i_sample}.h5ad`.')
                continue

            if i_mod == 'rna':

                if not 'rna' in self.modalities.keys(): self.modalities['rna'] = {}

                # we automatically infer from the given location names to select
                # the correct way of loading samples:

                if i_loc.endswith('.tsv.gz') or i_loc.endswith('.csv.gz') or \
                    i_loc.endswith('.tsv') or i_loc.endswith('.csv'):

                    self.modalities['rna'][i_sample] = read_table_rna(
                        src = i_loc, metadata = meta, sample = i_sample,
                        raw = False, default_taxa = i_taxa, eccentric = eccentric
                    )

                elif i_loc.endswith('.h5') or i_loc.endswith('.h5ad'):

                    self.modalities['rna'][i_sample] = read_h5ad_rna(
                        src = i_loc, metadata = meta, sample = i_sample,
                        raw = False, default_taxa = i_taxa
                    )

                else:
                    self.modalities['rna'][i_sample] = read_mtx_rna(
                        src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                        raw = False, default_taxa = i_taxa, eccentric = eccentric
                    )
                    
                self.modalities['rna'][i_sample].var = \
                    st.search_genes(self.modalities['rna'][i_sample].var_names.tolist())
                
                # in search for spliced and unspliced reads

                splicing = (
                    (meta.dataframe['sample'] == i_sample) & 
                    (meta.dataframe['modality'] == 'rna.splicing')
                )

                if splicing.sum() > 1:
                    warning(f'ignored spliced reads for sample [{i_sample}], since you specify more than one.')
                
                elif splicing.sum() == 1:
                    
                    import warnings
                    warnings.filterwarnings('ignore')

                    splice_loc = meta.dataframe.loc[splicing, :].iloc[0, 0]
                    if os.path.isdir(splice_loc):
                        self.modalities['rna'][i_sample] = attach_splice_reads_mtx(
                            self.modalities['rna'][i_sample], 
                            splice_loc, default_taxa = i_taxa,
                            sample = i_sample
                        )

                    elif splice_loc.endswith('.loom'):
                        self.modalities['rna'][i_sample] = attach_splice_reads_loom(
                            self.modalities['rna'][i_sample], 
                            splice_loc, default_taxa = i_taxa,
                            sample = i_sample
                        )
                    
                    else: error('skipped spliced matrix, not valid data format.')
                    warnings.filterwarnings('default')
                
                else: pass


            elif i_mod == 'rna.splicing': pass
            elif i_mod == 'rna.tcr':

                # 10x tcr folder
                tcr = parse_tcr_10x(
                    os.path.join(i_loc, 'filtered_contig_annotations.csv'),
                    sample = i_sample,
                    filter_non_productive = True,
                    filter_non_full_length = True
                )

                if not os.path.exists(os.path.join(self.directory, 'tcr')):
                    os.makedirs(os.path.join(self.directory, 'tcr'), exist_ok = True)

                tcr.to_csv(
                    os.path.join(self.directory, 'tcr', i_sample + '.tsv.gz'), 
                    sep = '\t', index = False
                )
            
            elif i_mod == 'rnasp-c':
                
                if not 'rnasp-c' in self.modalities.keys(): self.modalities['rnasp-c'] = {}

                from exprmat.reader.spatial import read_seekspace, read_xenium_explorer, is_xenium_explorer
                
                if is_xenium_explorer(i_loc):
                    self.modalities['rnasp-c'][i_sample] = read_xenium_explorer(
                        src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                        raw = False, default_taxa = i_taxa, eccentric = eccentric
                    )
                
                else:
                    self.modalities['rnasp-c'][i_sample] = read_seekspace(
                        src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                        raw = False, default_taxa = i_taxa, eccentric = eccentric
                    )

                self.modalities['rnasp-c'][i_sample].var = \
                    st.search_genes(self.modalities['rnasp-c'][i_sample].var_names.tolist())
                
            elif i_mod == 'rnasp-b':
                
                if not 'rnasp-b' in self.modalities.keys(): self.modalities['rnasp-b'] = {}

                from exprmat.reader.spatial import read_visium
                self.modalities['rnasp-b'][i_sample] = read_visium(
                    src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                    raw = False, default_taxa = i_taxa, eccentric = eccentric
                )

                self.modalities['rnasp-b'][i_sample].var = \
                    st.search_genes(self.modalities['rnasp-b'][i_sample].var_names.tolist())
                
            elif i_mod == 'rnasp-s':
                
                if not 'rnasp-s' in self.modalities.keys(): self.modalities['rnasp-s'] = {}

                from exprmat.reader.spatial import read_visium_hd
                self.modalities['rnasp-s'][i_sample], cellseg = read_visium_hd(
                    src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                    raw = False, default_taxa = i_taxa, eccentric = eccentric
                )

                self.modalities['rnasp-s'][i_sample].var = \
                    st.search_genes(self.modalities['rnasp-s'][i_sample].var_names.tolist())
                
                if cellseg is not None:
                    if not 'rnasp-c' in self.modalities.keys(): self.modalities['rnasp-c'] = {}
                    self.modalities['rnasp-c'][i_sample] = cellseg
                    self.modalities['rnasp-c'][i_sample].var = \
                        st.search_genes(self.modalities['rnasp-c'][i_sample].var_names.tolist())
            
            elif i_mod == 'atac':
                
                if not 'atac' in self.modalities.keys(): self.modalities['atac'] = {}
                default_assembly = i_assembly

                from exprmat.reader.peaks import import_fragments
                frags = import_fragments(
                    i_loc,
                    assembly = default_assembly,
                    sorted_by_barcode = False,
                )

                frags.uns['assembly'] = default_assembly
                # frags must not have the var table. otherwise error will occur when
                # assigning bins to the vars.
                self.modalities['atac'][i_sample] = frags

            elif i_mod in ['rna-bulk', 'atac-bulk']:
                
                # samples ended up with '-bulk' suffix will be processed into one
                # predefined anndata named 'bulk' in the corresoponding modality.
                # note that samples with different taxa property cannot be analysed together.
                
                pass

            else: warning(f'sample {i_sample} have no supported modalities')


        # attaching bulk datasets.

        for i_mod in ['rna-bulk', 'atac-bulk']:
            
            meta = self.metadata.dataframe.loc[
                self.metadata.dataframe['modality'] == i_mod, :
            ]

            if len(meta) == 0: continue

            if i_mod == 'rna-bulk':

                # bulk rna sequencing accepts expression matrices based on raw
                # counts (we recommend raw counts over any processed forms)
                # however, if normalized forms are only available, you may provide them.

                # it accepts generally any form of table, where the first row
                # contains gene names, and contains a row matched to sample name.

                datatable = []
                datarows = []
                example_row = None

                for sid in range(len(meta)):

                    prop = meta.iloc[sid]

                    if '/' in prop['taxa']: i_taxa, i_assembly = prop['taxa'].split('/')
                    else:
                        i_taxa = prop['taxa'] 
                        i_assembly = cfg['default.assembly'][i_taxa]

                    from exprmat.reader.matcher import convert_to_ugene
                    from exprmat.reader.spatial import read_table_from_extension

                    pdf = read_table_from_extension(prop['location'])
                    sampdf = pdf[[pdf.columns[0], prop['sample']]].set_index(pdf.columns[0])

                    names, gmask, notinlist = convert_to_ugene(sampdf.index.tolist(), i_taxa)
                    info(f'{len(notinlist)} unmatched genes from {len(sampdf)} rows.')
                    sampdf = sampdf.loc[gmask, :].copy()
                    sampdf.index = names
                    sampdf = sampdf.loc[~sampdf.index.duplicated(), :].copy()
                    datatable += [sampdf]
                    datarows += [pd.DataFrame(prop.copy()).T]
                    example_row = prop.copy()
                
                if len(datatable) == 1: datatable = datatable[0]
                elif len(datatable) == 2: datatable = datatable[0].join(datatable[1])
                else: datatable = datatable[0].join(datatable[1:])

                datarows = pd.concat(datarows)
                datarows.index = datatable.columns.tolist()

                # initialize anndata.
                adata = ad.AnnData(X = datatable.T, obs = datarows)
                adata.var = st.search_genes(adata.var_names.tolist())
                adata.obs_names = 'bulk-rna:' + adata.obs_names
                if not 'rna' in self.modalities.keys(): self.modalities['rna'] = {}
                self.modalities['rna']['bulk-rna'] = adata

                # replace metadata
                example_row['sample'] = 'bulk-rna'
                example_row['location'] = ':/rna/bulk-rna'
                example_row['group'] = '.'
                example_row['taxa'] = '.'
                example_row['modality'] = 'rna'
                example_row['batch'] = 'autogen'

                self.metadata.dataframe = self.metadata.dataframe.loc[
                    self.metadata.dataframe['modality'] != 'rna-bulk', :
                ].copy()
                self.metadata.insert_row(example_row)
            

            elif i_mod == 'atac-bulk':

                # bulk atac seq accepts bams as input.
                # if no bam files accessible, a bedgraph file can be supplied as an alternative
                # but several features (e.g. cut site identification) rely on raw fragments
                # and cannot be applied then.

                datarows = []
                if len(meta['taxa'].unique()) != 1:
                    error('failed to merge atac experiments with different reference.')
                
                i_taxa = meta['taxa'].unique().tolist()[0]
                if '/' in i_taxa: i_taxa, i_assembly = i_taxa.split('/')
                else: i_assembly = cfg['default.assembly'][i_taxa]

                example_row = None
                for sid in range(len(meta)):

                    prop = meta.iloc[sid].copy()

                    if prop['location'].endswith('.bampe'):
                        prop['bam'] = prop['location']
                    elif prop['location'].endswith('.bam'):
                        prop['bam'] = prop['location']
                    else: prop['bam'] = ''

                    if prop['location'].endswith('.bdg'):
                        prop['bedgraph'] = prop['location']
                    else: prop['bedgraph'] = ''

                    datarows += [pd.DataFrame(prop).T]
                    example_row = prop.copy()
                
                datarows = pd.concat(datarows)
                datarows.index = 'bulk-atac:' + datarows['sample']
                datarows.index.name = None
                from scipy.sparse import csr_matrix
                adata = ad.AnnData(
                    X = csr_matrix((len(datarows), 0), dtype = np.float32), 
                    obs = datarows
                )

                adata.uns['assembly'] = i_assembly
                from exprmat.data.finders import get_genome_size
                adata.uns['assembly.size'] = get_genome_size(i_assembly)

                if not 'atac' in self.modalities.keys(): self.modalities['atac'] = {}
                self.modalities['atac']['bulk-atac'] = adata

                # replace metadata
                example_row['sample'] = 'bulk-atac'
                example_row['location'] = ':/rna/bulk-atac'
                example_row['group'] = '.'
                example_row['taxa'] = '.'
                example_row['modality'] = 'atac'
                example_row['batch'] = 'autogen'

                self.metadata.dataframe = self.metadata.dataframe.loc[
                    self.metadata.dataframe['modality'] != 'atac-bulk', :
                ].copy()
                self.metadata.insert_row(example_row)

        pass


    @property
    def obs(self):
        self.check_merged()
        self.pull()
        return self.mudata.obs
    

    @property
    def var(self):
        self.check_merged()
        return self.mudata.var
    

    @property
    def shape(self):
        self.check_merged()
        return (self.mudata.n_obs, self.mudata.n_vars)
    
    
    def merge(
        self, join = 'outer', obsms = [], variable_columns = [],
        bool_merge_behavior = 'or',
        numeric_merge_behavior = 'mean',
        string_merge_behavior = 'unique_concat', string_merge_sep = ';',
        subset_dict = None, subset_key = 'leiden'
    ):
        '''
        Merge the separate modalities, and generate the integrated dataset.
        Note that this integration is merely concatenating the raw matrices without
        any batch correction. You should run batch correction if needed using the
        routines elsewhere. (or the general interface ``integrate(...)``).
        '''

        # the var names are self-interpretable, and we will merge the samples
        # and throw away original column metadata. for atac-seq experiments, however,
        # the original var metadata is useful, we should store them and append
        # to the merged dataset later.

        merged = {}

        if 'rna' in self.modalities.keys():

            filtered = {}
            for rnak in self.modalities['rna'].keys():

                # if following the recommended routine, by the time one will need
                # to merge the datasets, the X slot should contain log normalized values.

                filtered[rnak] = ad.AnnData(
                    X = self.modalities['rna'][rnak].X,
                    obs = self.modalities['rna'][rnak].obs,
                    var = self.modalities['rna'][rnak].var
                )

                if 'counts' in self.modalities['rna'][rnak].layers.keys():
                    filtered[rnak].layers['counts'] = self.modalities['rna'][rnak].layers['counts']

                if 'spliced' in self.modalities['rna'][rnak].layers.keys():
                    filtered[rnak].layers['spliced'] = self.modalities['rna'][rnak].layers['spliced']

                if 'unspliced' in self.modalities['rna'][rnak].layers.keys():
                    filtered[rnak].layers['unspliced'] = self.modalities['rna'][rnak].layers['unspliced']

                if 'ambiguous' in self.modalities['rna'][rnak].layers.keys():
                    filtered[rnak].layers['ambiguous'] = self.modalities['rna'][rnak].layers['ambiguous']
                
                for obsm in obsms:
                    filtered[rnak].obsm[obsm] = self.modalities['rna'][rnak].obsm[obsm]

                if subset_dict is not None:
                    if rnak not in subset_dict.keys():
                        del filtered[rnak]
                    
                    else:
                        cell_mask = [
                            x in subset_dict[rnak] 
                            for x in filtered[rnak].obs[subset_key].tolist()
                        ]

                        if not all(cell_mask):
                            filtered[rnak] = filtered[rnak][cell_mask, :]


            # merge rna st.
            merged['rna'] = ad.concat(
                filtered, axis = 'obs', 
                join = join, label = 'sample'
            )

            # retrieve the corresponding gene info according to the universal 
            # nomenclature rna:[tax]:[ugene] format

            gene_names = merged['rna'].var_names.tolist()
            merged['rna'].var = st.search_genes(gene_names)

            # we will next merge variable columns. this is typically designed for
            # merging bool vector masks for hvgs.
            
            for varc in variable_columns:
                
                values = {}

                for rnak in self.modalities['rna'].keys():

                    # just skip samples with explicitly ignored subset.
                    if subset_dict is not None:
                        if rnak not in subset_dict.keys():
                            continue

                    key = self.modalities['rna'][rnak].var.index.tolist()
                    if not varc in self.modalities['rna'][rnak].var.columns.tolist():
                        warning(f'sample `{rnak}` does not contain variable column `{key}`. skipped.')
                        continue

                    value = self.modalities['rna'][rnak].var[varc].tolist()
                    for idk in range(len(key)):
                        if key[idk] not in values.keys(): values[key[idk]] = []
                        values[key[idk]] += [value[idk]]
                
                labels = values[list(values.keys())[0]]
                if type(labels[0]) is str:

                    if string_merge_behavior == 'unique_concat':
                        f = lambda l: string_merge_sep.join(list(set(l)))
                    elif string_merge_behavior == 'concat':
                        f = lambda l: string_merge_sep.join(l)
                    else: f = lambda l: 'NA'

                    merged_var = [f(values[g]) if g in values.keys() else 'NA' for g in gene_names]
                    merged['rna'].var[varc] = merged_var

                elif (type(labels[0]) is int) or \
                     (type(labels[0]) is float) or \
                     (type(labels[0]) is np.float32) or \
                     (type(labels[0]) is np.float64) or \
                     (type(labels[0]) is np.int32):

                    if numeric_merge_behavior == 'mean':
                        f = lambda l: np.mean(l)
                    elif string_merge_behavior == 'var':
                        f = lambda l: np.var(l)
                    elif string_merge_behavior == 'sd':
                        f = lambda l: np.std(l)
                    elif string_merge_behavior == 'max':
                        f = lambda l: np.max(l)
                    elif string_merge_behavior == 'min':
                        f = lambda l: np.min(l)
                    elif string_merge_behavior == 'median':
                        f = lambda l: np.median(l)
                    else: f = lambda l: float('nan')

                    merged_var = [f(values[g]) if g in values.keys() else float('nan') for g in gene_names]
                    merged['rna'].var[varc] = merged_var
                
                elif type(labels[0]) is bool:

                    if bool_merge_behavior == 'and':
                        f = lambda l: all(l)
                    elif bool_merge_behavior == 'or':
                        f = lambda l: any(l)
                    else: f = lambda l: False

                    merged_var = [f(values[g]) if g in values.keys() else False for g in gene_names]
                    merged['rna'].var[varc] = merged_var

                else: warning(f'`{key}` with unsupported type. skipped.')
                pass

            pass # merging 'rna'.
        
        
        # integrate tcr metadata
        if ('rna' in merged.keys()) and os.path.exists(os.path.join(self.directory, 'tcr')):
            self.rna_attach_tcr(merged['rna'], os.path.join(self.directory, 'tcr'))

        if 'atac' in self.modalities.keys():
            
            filtered = {}
            assemblies = []
            assembly_size = None
            nvs = []
            var_table = None

            for atack in self.modalities['atac'].keys():
                
                filtered[atack] = ad.AnnData(
                    X = self.modalities['atac'][atack].X,
                    obs = self.modalities['atac'][atack].obs,
                    var = self.modalities['atac'][atack].var,
                )

                nvs.append(self.modalities['atac'][atack].n_vars)
                var_table = self.modalities['atac'][atack].var
                filtered[atack].uns['assembly'] = self.modalities['atac'][atack].uns['assembly'] 
                filtered[atack].uns['assembly.size'] = self.modalities['atac'][atack].uns['assembly.size'] 
                assemblies.append(self.modalities['atac'][atack].uns['assembly'] )
                assembly_size = self.modalities['atac'][atack].uns['assembly.size']

                if 'paired' in self.modalities['atac'][atack].obsm.keys():
                    filtered[atack].obsm['paired'] = self.modalities['atac'][atack].obsm['paired'] 
                if 'single' in self.modalities['atac'][atack].obsm.keys():
                    filtered[atack].obsm['single'] = self.modalities['atac'][atack].obsm['single'] 
            
            # check the assembly to be completely identical
            if len(set(assemblies)) != 1:
                warning(f'cannot merge atac data with different genomic assemblies.')
            
            if len(set(nvs)) != 1:
                warning(f'cannot merge atac data with inconsistant bins.')
                warning(f'you should not filter out the bins (or select features) before integration.')
            
            else:
                merged['atac'] = ad.concat(
                    filtered, axis = 'obs', 
                    join = join, label = 'sample'
                )

                merged['atac'].uns['assembly'] = assemblies[0]
                merged['atac'].uns['assembly.size'] = assembly_size
                if var_table is not None:
                    merged['atac'].var = var_table.loc[
                        merged['atac'].var_names, 
                        ['.seqid', '.start', '.end', 'location', 'unique']
                    ]

            pass

        if 'rnasp-c' in self.modalities.keys():

            filtered = {}
            spatialdict = {}

            for rnak in self.modalities['rnasp-c'].keys():

                # if following the recommended routine, by the time one will need
                # to merge the datasets, the X slot should contain log normalized values.

                filtered[rnak] = ad.AnnData(
                    X = self.modalities['rnasp-c'][rnak].X,
                    obs = self.modalities['rnasp-c'][rnak].obs,
                    var = self.modalities['rnasp-c'][rnak].var,
                )

                if 'counts' in self.modalities['rnasp-c'][rnak].layers.keys():
                    filtered[rnak].layers['counts'] = self.modalities['rnasp-c'][rnak].layers['counts']

                if 'spliced' in self.modalities['rnasp-c'][rnak].layers.keys():
                    filtered[rnak].layers['spliced'] = self.modalities['rnasp-c'][rnak].layers['spliced']

                if 'unspliced' in self.modalities['rnasp-c'][rnak].layers.keys():
                    filtered[rnak].layers['unspliced'] = self.modalities['rnasp-c'][rnak].layers['unspliced']

                if 'ambiguous' in self.modalities['rnasp-c'][rnak].layers.keys():
                    filtered[rnak].layers['ambiguous'] = self.modalities['rnasp-c'][rnak].layers['ambiguous']
                
                for obsm in list(set(['spatial'] + obsms)):
                    filtered[rnak].obsm[obsm] = self.modalities['rnasp-c'][rnak].obsm[obsm]

                for spsample in self.modalities['rnasp-c'][rnak].uns['spatial'].keys():
                    spatialdict[spsample] = self.modalities['rnasp-c'][rnak].uns['spatial'][spsample]

                if subset_dict is not None:
                    if rnak not in subset_dict.keys():
                        del filtered[rnak]
                    
                    else:
                        cell_mask = [
                            x in subset_dict[rnak] 
                            for x in filtered[rnak].obs[subset_key].tolist()
                        ]

                        if not all(cell_mask):
                            filtered[rnak] = filtered[rnak][cell_mask, :]


            # merge rna st.
            merged['rnasp-c'] = ad.concat(
                filtered, axis = 'obs', 
                join = join, label = 'sample'
            )

            merged['rnasp-c'].uns['spatial'] = spatialdict

            # retrieve the corresponding gene info according to the universal 
            # nomenclature rna:[tax]:[ugene] format

            gene_names = merged['rnasp-c'].var_names.tolist()
            merged['rnasp-c'].var = st.search_genes(gene_names)

            # we will next merge variable columns. this is typically designed for
            # merging bool vector masks for hvgs.
            
            for varc in variable_columns:
                
                values = {}

                for rnak in self.modalities['rnasp-c'].keys():

                    # just skip samples with explicitly ignored subset.
                    if subset_dict is not None:
                        if rnak not in subset_dict.keys():
                            continue

                    key = self.modalities['rnasp-c'][rnak].var.index.tolist()
                    if not varc in self.modalities['rnasp-c'][rnak].var.columns.tolist():
                        warning(f'sample `{rnak}` does not contain variable column `{key}`. skipped.')
                        continue

                    value = self.modalities['rnasp-c'][rnak].var[varc].tolist()
                    for idk in range(len(key)):
                        if key[idk] not in values.keys(): values[key[idk]] = []
                        values[key[idk]] += [value[idk]]
                
                labels = values[list(values.keys())[0]]
                if type(labels[0]) is str:

                    if string_merge_behavior == 'unique_concat':
                        f = lambda l: string_merge_sep.join(list(set(l)))
                    elif string_merge_behavior == 'concat':
                        f = lambda l: string_merge_sep.join(l)
                    else: f = lambda l: 'NA'

                    merged_var = [f(values[g]) if g in values.keys() else 'NA' for g in gene_names]
                    merged['rna'].var[varc] = merged_var

                elif (type(labels[0]) is int) or \
                     (type(labels[0]) is float) or \
                     (type(labels[0]) is np.float32) or \
                     (type(labels[0]) is np.float64) or \
                     (type(labels[0]) is np.int32):

                    if numeric_merge_behavior == 'mean':
                        f = lambda l: np.mean(l)
                    elif string_merge_behavior == 'var':
                        f = lambda l: np.var(l)
                    elif string_merge_behavior == 'sd':
                        f = lambda l: np.std(l)
                    elif string_merge_behavior == 'max':
                        f = lambda l: np.max(l)
                    elif string_merge_behavior == 'min':
                        f = lambda l: np.min(l)
                    elif string_merge_behavior == 'median':
                        f = lambda l: np.median(l)
                    else: f = lambda l: float('nan')

                    merged_var = [f(values[g]) if g in values.keys() else float('nan') for g in gene_names]
                    merged['rnasp-c'].var[varc] = merged_var
                
                elif type(labels[0]) is bool:

                    if bool_merge_behavior == 'and':
                        f = lambda l: all(l)
                    elif bool_merge_behavior == 'or':
                        f = lambda l: any(l)
                    else: f = lambda l: False

                    merged_var = [f(values[g]) if g in values.keys() else False for g in gene_names]
                    merged['rnasp-c'].var[varc] = merged_var

                else: warning(f'`{key}` with unsupported type. skipped.')
                pass

            pass # merging 'rna'.


        if len(merged) > 0:
            mdata = mu.MuData(merged)
            self.mudata = mdata

        else: self.mudata = None


    def do_for(self, modality, samples, func, **kwargs):
        
        results = {}
        for mod, samp in zip(
            self.metadata.dataframe['modality'].tolist(),
            self.metadata.dataframe['sample'].tolist()
        ):
            if '.' in mod: continue
            if mod != modality: continue

            do = False
            if samples is None: do = True
            if isinstance(samples, list):
                if samp in samples: do = True

            if do:
                
                results[samp] = None

                if self.modalities is None:
                    warning(f'experiment do not load any samples.')
                    return
                
                if not mod in self.modalities.keys():
                    warning(f'{mod} does not loaded in the modalities key.')
                    continue
                if not samp in self.modalities[mod].keys():
                    warning(f'{samp} not loaded in the {mod} modality.')
                    continue
                
                # try: results[samp] = func(self.modalities[mod][samp], samp, **kwargs)
                # except: warning(f'method failed for sample {samp}')
                results[samp] = func(self.modalities[mod][samp], samp, **kwargs)
        
        return results


    def do_for_modality(self, modality, run_on_samples, func, **kwargs):
        if isinstance(run_on_samples, bool) and run_on_samples:
            return self.do_for(modality, self.all_samples(modality), func, **kwargs)
        elif isinstance(run_on_samples, list):
            return self.do_for(
                modality, list(set(self.all_samples(modality)) & set(run_on_samples)), 
                func, **kwargs
            )
        
        else:
            assert modality in self.mudata.mod.keys()
            return func(self.mudata[modality], 'integrated', **kwargs)
        
    def do_for_rna(self, run_on_samples, func, **kwargs):
        return self.do_for_modality('rna', run_on_samples, func, **kwargs)
    
    def do_for_atac(self, run_on_samples, func, **kwargs):
        return self.do_for_modality('atac', run_on_samples, func, **kwargs)
    
    def do_for_atac_peaks(self, run_on_samples, func, **kwargs):
        return self.do_for_modality('atac-p', run_on_samples, func, **kwargs)
    
    def do_for_atac_gene_activity(self, run_on_samples, func, **kwargs):
        return self.do_for_modality('atac-g', run_on_samples, func, **kwargs)
    
    def do_for_rnaspc(self, run_on_samples, func, **kwargs):
        return self.do_for_modality('rnasp-c', run_on_samples, func, **kwargs)
    
    def do_for_rnaspb(self, run_on_samples, func, **kwargs):
        return self.do_for_modality('rnasp-b', run_on_samples, func, **kwargs)
        
    
    def plot_for_modality(
        self, modality, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        from exprmat.utils import setup_styles
        setup_styles()

        if isinstance(run_on_samples, bool) and run_on_samples:
            figures = self.do_for(modality, self.all_samples(modality), func, **kwargs)
            for f in figures.values(): f.tight_layout()
            return figures
        
        elif isinstance(run_on_samples, list):
            figures = self.do_for(
                modality, list(set(self.all_samples(modality)) & set(run_on_samples)), 
                func, **kwargs
            )
            
            for f in figures.values(): f.tight_layout()
            return figures
        
        else:

            # the merged modality. this is not restricted to the sample table,
            # and we may generate secondary (artifical) modality that is not
            # directly loaded from disks.

            assert modality in self.mudata.mod.keys()

            if not run_on_splits:
                figure = func(self.mudata[modality], 'integrated', **kwargs)
                import warnings
                warnings.filterwarnings('ignore')
                figure.tight_layout()
                return figure

            else:
                results = {}
                cats = self.mudata[modality].obs[split_key].cat.categories.tolist()
                if split_selection is None: split_selection = cats

                n_features = len(split_selection)
                for feat_id in range(len(split_selection)):

                    results[split_selection[feat_id]] = func(
                        self.mudata[modality][
                            self.mudata[modality].obs[split_key] == split_selection[feat_id],:].copy(),
                        # copy the data to silence the implicit modification warning on views. 
                        split_selection[feat_id], 
                        **kwargs
                    )

                    results[split_selection[feat_id]].tight_layout()
                
                return results
    
    def plot_for_rna(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        return self.plot_for_modality(
            'rna', run_on_samples, func, 
            run_on_splits, split_key, split_selection, **kwargs
        )

    def plot_for_atac(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        return self.plot_for_modality(
            'atac', run_on_samples, func, 
            run_on_splits, split_key, split_selection, **kwargs
        )

    def plot_for_atac_peaks(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        return self.plot_for_modality(
            'atac-p', run_on_samples, func, 
            run_on_splits, split_key, split_selection, **kwargs
        )

    def plot_for_atac_gene_activity(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        return self.plot_for_modality(
            'atac-g', run_on_samples, func, 
            run_on_splits, split_key, split_selection, **kwargs
        )
    
    def plot_for_rnaspc(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        return self.plot_for_modality(
            'rnasp-c', run_on_samples, func, 
            run_on_splits, split_key, split_selection, **kwargs
        )
    
    def plot_for_rnaspb(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        return self.plot_for_modality(
            'rnasp-b', run_on_samples, func, 
            run_on_splits, split_key, split_selection, **kwargs
        )

    
    def all_samples(self, modality = None):
        if modality is None:
            return self.metadata.dataframe['sample'].tolist()
        elif isinstance(modality, str): 
            return self.metadata.dataframe[
                (self.metadata.dataframe['modality'] == modality)
            ]['sample'].tolist()
        elif isinstance(modality, list):
            return self.metadata.dataframe.loc[
                [x in modality for x in self.metadata.dataframe['modality']], :
            ]['sample'].tolist()
    

    def all_rna_samples(self):
        return self.all_samples(['rna', 'atac.g'])
    

    def all_atac_samples(self):
        return self.all_samples('atac')
    

    def check_merged(self, slot = None):
        if self.mudata is None:
            error('requires the dataset to be merged before running this method.') 
        if slot is not None:
            if slot not in self.mudata.mod.keys():
                error(f'requires specific slot `{slot}` be in merged mudata.') 

    
    def build_subset(
        self, subset_name, slot = 'rna',
        values = None, keys = ['sample']
    ):
        self.check_merged(slot)
        self.pull()

        masks = []
        final = np.array([True] * self.mudata.n_obs, dtype = np.bool)
        for idx in range(len(keys)):
            masks += [np.array([x in values[idx] for x in self.mudata.obs[f'{slot}:{keys[idx]}'].tolist()])]
        
        for mask in masks: final = final & mask
        subset = self.mudata[final, :].copy()
        info(f'selected {final.sum()} observations from {len(final)}.')

        if not os.path.exists(os.path.join(self.directory, 'subsets')):
            os.makedirs(os.path.join(self.directory, 'subsets'))
        
        subset.write_h5mu(os.path.join(self.directory, 'subsets', subset_name + '.h5mu'))


    def build_subsample(
        self, subset_name, slot = 'rna',
        key = None, n = 3000
    ):
        self.check_merged(slot)
        self.pull()
        from exprmat.dynamics.utils import indices_to_bool

        if key is None:
            if n > self.mudata.n_obs:
                n = self.mudata.n_obs
                warning('subsample n > obs. returning all indices')
            else: info(f'randomly selects {n} cells from {self.mudata.n_obs}.')
            rind = np.random.choice(self.mudata.n_obs, size = n, replace = False)
            mask = indices_to_bool(rind, self.mudata.n_obs)

        elif isinstance(n, dict):
            values = self.mudata.obs[f'{slot}:{key}'].unique().tolist()
            rind = np.array([])
            for val in values:
                indices = np.argwhere(self.mudata.obs[f'{slot}:{key}'] == val).flatten()
                if str(val) in n.keys(): _n = n[str(val)]
                else: continue
                if _n == '*': _n = len(indices)

                if _n > len(indices):
                    _n = len(indices)
                    warning(f'subsample n > len({val}). returning all indices')
                else: info(f'randomly selects {_n} cells from [{val}] ({len(indices)}).')
                rind = np.append(rind, np.random.choice(indices, size = _n, replace = False))
            mask = indices_to_bool(rind, self.mudata.n_obs)

        subset = self.mudata[mask, :].copy()
        info(f'selected {mask.sum()} observations from {len(mask)}.')

        if not os.path.exists(os.path.join(self.directory, 'subsets')):
            os.makedirs(os.path.join(self.directory, 'subsets'))
        
        subset.write_h5mu(os.path.join(self.directory, 'subsets', subset_name + '.h5mu'))


    def annotate(
        self, slot = 'rna', annotation = 'cell.type',
        mapping = {}, cluster = 'leiden'
    ):
        self.check_merged(slot)

        reverse_dict = {}
        for key in mapping.keys():
            value = mapping[key]
            if isinstance(value, int):
                reverse_dict[str(value)] = key
            elif isinstance(value, str):
                reverse_dict[value] = key
            elif isinstance(value, list):
                for l in value:
                    reverse_dict[str(l)] = key
            
        annot = [
            reverse_dict[x] if x in reverse_dict.keys()
            else x
            for x in self.mudata[slot].obs[cluster].tolist()
        ]
        
        self.mudata[slot].obs[annotation] = annot
        self.mudata[slot].obs[annotation] = self.mudata[slot].obs[annotation].astype('category')
        print(self.mudata[slot].obs[annotation].value_counts())


    def annotate_merge(
        self, slot = 'rna', annotation = 'cell.type',
        merge = [], into = '_'
    ):
        self.check_merged(slot)
        ls = self.mudata[slot].obs[annotation].tolist()
        ls = [into if x in merge else x for x in ls]
        self.mudata[slot].obs[annotation] = ls
        self.mudata[slot].obs[annotation] = self.mudata[slot].obs[annotation].astype('category')
        print(self.mudata[slot].obs[annotation].value_counts())

    
    def annotate_paste(
        self, froms, into, slot = 'rna', sep = '.'
    ):
        self.check_merged(slot)
        ls = [self.mudata[slot].obs[f].tolist() for f in froms]
        concat = []
        for tupl in zip(*ls):
            concat.append(sep.join(list(tupl)))
        self.mudata[slot].obs[into] = concat
        self.mudata[slot].obs[into] = self.mudata[slot].obs[into].astype('category')
        print(self.mudata[slot].obs[into].value_counts())

    
    def annotate_broadcast(self, slot = 'rna', annotation = 'cell.type', todf = None, to = 'cell.type'):
        
        import h5py
        from anndata.io import read_elem, write_elem
        
        if todf is None:
            
            # broadcast to the same dataset (just a copy of non-nan values.)
            toslot = self.mudata[slot].obs[to].tolist() \
                if to in self.mudata[slot].obs.columns \
                    else [None] * self.mudata[slot].n_obs
            fromslot = self.mudata[slot].obs[annotation].tolist()
            self.mudata[slot].obs[to] = [
                x if str(x) != 'nan' else y 
                for x, y in zip(fromslot, toslot)
            ]

            info(f'merged {annotation} to {to} in current dataset.')
            print(self.mudata[slot].obs[to].value_counts())
            self.mudata[slot].obs[to] = self.mudata[slot].obs[to].astype('category')

        # if todf is another subset.
        elif isinstance(todf, str) and os.path.exists(
            os.path.join(self.directory, 'subsets', todf + '.h5mu')
        ):
            if (self.subset is not None) and (todf == self.subset):

                # broadcast to the same dataset (just a copy of non-nan values.)
                toslot = self.mudata[slot].obs[to].tolist() \
                    if to in self.mudata[slot].obs.columns \
                        else [None] * self.mudata[slot].n_obs
                fromslot = self.mudata[slot].obs[annotation].tolist()
                self.mudata[slot].obs[to] = [
                    x if str(x) != 'nan' else y 
                    for x, y in zip(fromslot, toslot)
                ]

                info(f'merged {annotation} to {to} in current dataset.')
                print(self.mudata[slot].obs[to].value_counts())
                self.mudata[slot].obs[to] = self.mudata[slot].obs[to].astype('category')

            else:
                
                with h5py.File(os.path.join(self.directory, 'subsets', todf + '.h5mu'), 'r+') as h5f:
                    
                    target_df = read_elem(h5f['mod']['rna']['obs'])

                    toslot = target_df[to].tolist() \
                        if to in target_df.columns \
                            else [None] * len(target_df)

                    fromtable = pd.DataFrame({
                        'index': self.mudata[slot].obs_names.tolist(),
                        '.temp': self.mudata[slot].obs[annotation].tolist()
                    })

                    fromtable = fromtable.set_index('index')
                    target_df = target_df.join(fromtable, how = 'left')

                    fromslot = target_df['.temp'].tolist()
                    target_df[to] = [
                        x if str(x) != 'nan' else y 
                        for x, y in zip(fromslot, toslot)
                    ]

                    del target_df['.temp']
                    info(f'updated {annotation} to {to} in subsets/{todf}.h5mu')
                    print(target_df[to].value_counts())
                    target_df[to] = target_df[to].astype('category')

                    write_elem(h5f, '/mod/rna/obs', target_df)
        

        elif isinstance(todf, str) and todf == 'integrated':
            
            with h5py.File(os.path.join(self.directory, 'integrated.h5mu'), 'r+') as h5f:
                    
                target_df = read_elem(h5f['mod']['rna']['obs'])

                toslot = target_df[to].tolist() \
                    if to in target_df.columns \
                        else [None] * len(target_df)

                fromtable = pd.DataFrame({
                    'index': self.mudata[slot].obs_names.tolist(),
                    '.temp': self.mudata[slot].obs[annotation].tolist()
                })

                fromtable = fromtable.set_index('index')
                target_df = target_df.join(fromtable, how = 'left')

                fromslot = target_df['.temp'].tolist()
                target_df[to] = [
                    x if str(x) != 'nan' else y 
                    for x, y in zip(fromslot, toslot)
                ]

                del target_df['.temp']
                info(f'updated {annotation} to {to} in {todf}.h5mu')
                print(target_df[to].value_counts())
                target_df[to] = target_df[to].astype('category')

                write_elem(h5f, '/mod/rna/obs', target_df)
            
            
    def exclude(
        self, slot = 'rna', annotation = 'cell.type', remove = []
    ):
        self.check_merged(slot)
        ls = self.mudata[slot].obs[annotation].tolist()
        mask = [x not in remove for x in ls]
        orig = self.mudata.n_obs
        self.mudata = self.mudata[mask, :].copy()
        info(f'keep {self.mudata.n_obs} observations from {orig}.')


    def exclude_outlier(
        self, slot = 'rna', embedding = 'umap', xlim = (-5, 5), ylim = (-5, 5)
    ):
        self.check_merged(slot)
        assert embedding in self.mudata[slot].obsm
        assert self.mudata[slot].obsm[embedding].shape[1] == 2
        mask = (
            (self.mudata[slot].obsm[embedding][:, 0] >= xlim[0]) &
            (self.mudata[slot].obsm[embedding][:, 0] <= xlim[1]) &
            (self.mudata[slot].obsm[embedding][:, 1] >= ylim[0]) &
            (self.mudata[slot].obsm[embedding][:, 1] <= ylim[1])
        )

        orig = self.mudata.n_obs
        self.mudata = self.mudata[mask, :]
        info(f'keep {self.mudata.n_obs} observations from {orig}.')


    def keep(
        self, slot = 'rna', annotation = 'cell.type', k = []
    ):
        self.check_merged(slot)
        ls = self.mudata[slot].obs[annotation].tolist()
        mask = [x in k for x in ls]
        orig = self.mudata.n_obs
        self.mudata = self.mudata[mask, :]
        info(f'keep {self.mudata.n_obs} observations from {orig}.')

    
    def exclude_nonexisting_variables(self, slot = 'rna'):
        # remove genes that do not express (since we make a subset of the total dataset).
        # pca doesn't allow columns that are completely made up of zeros.
        import numpy as np
        gene_mask = self.mudata.mod[slot].X.sum(axis = 0) <= 0.01
        print(f'removed {gene_mask.sum()} genes with zero expression.')
        self.mudata.mod[slot] = self.mudata.mod[slot][:, ~ gene_mask].copy()


    def find_variable(self, gene_name, slot = 'rna', layer = 'X'):
        self.check_merged(slot)
        from exprmat.utils import find_variable as fvar
        return fvar(self.mudata.mod[slot], gene_name = gene_name, layer = layer)
    

    def filter(self, into, conditions = [], criteria = [], slot = 'rna'):

        bool_masks = []
        assert len(conditions) == len(criteria)
        for cond, crit in zip(conditions, criteria):
            bool_masks.append(crit(self.find_variable(cond, slot)))
        
        npbool = bool_masks[0]
        if len(bool_masks) > 1:
            for bx in range(1, len(bool_masks)):
                npbool = npbool & bool_masks[bx]
        
        self.mudata[slot].obs[into] = ['true' if x else 'false' for x in npbool.tolist()]
        self.mudata[slot].obs[into] = self.mudata[slot].obs[into].astype('category')
        print(self.mudata[slot].obs[into].value_counts())

    
    # wrapper functions

    def run_rna_qc(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_qc, **kwargs)
        
    def run_rna_filter(self, run_on_samples = False):
        results = self.do_for_rna(run_on_samples, st.rna_filter)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_rna_filter_row_by_sum(self, run_on_samples = False, **kwargs):
        results = self.do_for_rna(run_on_samples, st.adata_filter_row_by_sum, **kwargs)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_rna_filter_column_by_sum(self, run_on_samples = False, **kwargs):
        results = self.do_for_rna(run_on_samples, st.adata_filter_column_by_sum, **kwargs)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_rna_log_normalize(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_log_normalize, **kwargs)

    def run_rna_select_hvg(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_select_hvg, **kwargs)

    def run_rna_scale_pca(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_scale_pca, **kwargs)

    def run_rna_knn(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_knn, **kwargs)

    def run_rna_leiden(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_leiden, **kwargs)

    def run_rna_leiden_subcluster(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_leiden_subcluster, **kwargs)

    def run_rna_umap(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_umap, **kwargs)

    def run_rna_mde(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_mde, **kwargs)

    def run_rna_mde_fit(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_mde_fit, **kwargs)

    def run_rna_integrate(self, method = 'harmony', dest = 'harmony', **kwargs):
        
        self.check_merged('rna')
        if method == 'harmony':
            from exprmat.preprocessing.integrate import harmony
            harmony(self.mudata['rna'], key = 'batch', adjusted_basis = dest, **kwargs)
        
        elif method == 'scanorama':
            from exprmat.preprocessing.integrate import scanorama
            scanorama(self.mudata['rna'], key = 'batch', adjusted_basis = dest, **kwargs)

        else: error(f'unsupported integration method `{method}`.')

    def run_rna_markers(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_markers, **kwargs)

    def run_rna_markers_deseq(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_markers_deseq, **kwargs)

    def run_rna_kde(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_kde, **kwargs)

    def run_rna_proportion(self, run_on_samples = False, **kwargs):
        '''
        This is a simplified method of cell type proportion calculation.
        It is implemented in earlier versions of the package and can be replaced by a more
        general version of counting summary. This returns a simple dataframe, while summary
        returns an annotated object and can be further processed using routines under
        ``exprmat.clustering.summary`` package.
        '''
        return self.do_for_rna(run_on_samples, st.rna_proportion, **kwargs)
    
    def run_rna_infercnv(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_infercnv, **kwargs)
    
    def run_rna_summary(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, st.rna_summary, **kwargs)
    
    def run_rna_aggregate(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, st.rna_aggregate, **kwargs)
    
    def run_rna_attach_tcr(self, run_on_samples = False):
        self.do_for_rna(
            run_on_samples, 
            st.rna_attach_tcr, 
            searchdir = os.path.join(self.directory, 'tcr')
        )

    def run_rna_calculate_tcr_metrics(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_calculate_tcr_metrics, **kwargs)
    
    def run_rna_aggregate_tcr_by_identity(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_aggregate_tcr_by_identity, **kwargs)
    
    def run_rna_calculate_startracs_metrics(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_calculate_startracs_metrics, **kwargs)
    
    def run_rna_calculate_startracs_pairwise_metrics(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_calculate_startracs_pairwise_metrics, **kwargs)
    
    def run_rna_expression_mask(
        self, run_on_samples = False, gene = None, key = 'mask', 
        lognorm = 'X', threshold = 0.1, negate = False
    ):
        self.do_for_rna(
            run_on_samples, st.rna_expression_mask, 
            gene = gene, key = key, lognorm = lognorm, threshold = threshold,
            negate = negate
        )
    
    def run_rna_gsea(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_gsea, **kwargs)
    
    def run_rna_opa(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_opa, **kwargs)
    
    def run_rna_gsva(self, run_on_samples = False, key_added = 'gsva', **kwargs):
        gsv = self.do_for_rna(run_on_samples, st.rna_gsva, **kwargs)
        if not run_on_samples:
            gsv.var['gset'] = gsv.var_names.tolist()
            gsv.var_names = [key_added + ':' + str(i + 1) for i in range(gsv.n_vars)]
            self.mudata.mod[key_added] = gsv
        return gsv
    
    def run_rna_remove_slots(self, run_on_samples = False, slot = 'obs', names = []):
        self.do_for_rna(run_on_samples, st.remove_slot, slot = slot, names = names)
    
    def run_rna_ligand_receptor(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_ligand_receptor, **kwargs)
    
    def run_rna_score_genes(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, st.rna_score_genes, **kwargs)
    
    def run_rna_score_genes_gsva(self, run_on_samples = False, key_added = 'gsva.scores', **kwargs):
        gsv = self.do_for_rna(run_on_samples, st.rna_score_genes_gsva, **kwargs)
        if not run_on_samples:
            gsv.var['gset'] = gsv.var_names.tolist()
            gsv.var_names = [key_added + ':' + str(i + 1) for i in range(gsv.n_vars)]
            self.mudata.mod[key_added] = gsv
        return gsv
    
    def run_rna_velocity(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_velocity, **kwargs)
    
    def run_rna_consensus_nmf(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_consensus_nmf, **kwargs)
    
    def run_rna_consensus_nmf_extract_k(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_consensus_nmf_extract_k, **kwargs)
    
    def run_rna_impute_magic(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, st.rna_impute_magic, **kwargs) 
    
    def run_atac_make_bins(self, run_on_samples = False, **kwargs):
        # bins can be made directly from bedgraph files if there is no alignment bam provided.
        # thus these samples can be used for clustering.
        self.do_for_atac(run_on_samples, st.atac_make_bins, **kwargs)
    
    def run_atac_filter_cells(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.atac_filter_cells, **kwargs)
    
    def run_atac_select_features(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.atac_select_features, **kwargs)
    
    def run_atac_spectral(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.atac_spectral, **kwargs)
    
    def run_atac_scrublet(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.atac_scrublet, **kwargs)
    
    def run_atac_knn(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.rna_knn, **kwargs)
    
    def run_atac_umap(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.rna_umap, **kwargs)
    
    def run_atac_leiden(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.rna_leiden, **kwargs)
    
    def run_atac_leiden_subcluster(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.rna_leiden_subcluster, **kwargs)
    
    def run_atac_infer_gene_activity(self, run_on_samples = False, **kwargs):
        # inference on gene activity provides two alternative way. one from bam files
        # in the fragment matrix, and another from bedgraph counting.
        # more efficiently, should count on bigwig files.

        if not run_on_samples:
            if 'atac.g' in self.mudata.mod.keys():
                warning('atac.g modality exists in the mudata object. the run is cancelled to prevent overwrite.')
                error('if you want to overwrite existing modality, you should delete it manually first.')

        data = self.do_for_atac(run_on_samples, st.atac_infer_gene_activity, **kwargs)
        if isinstance(data, dict):
            for sampkey in data.keys():
                prop = self.metadata.dataframe.loc[
                    (self.metadata.dataframe['sample'] == sampkey) &
                    (self.metadata.dataframe['modality'] == 'atac'), :
                ]

                assert len(prop) == 1
                prop = prop.iloc[0, :].copy()
                prop['modality'] = 'atac-g'
                if not 'atac-g' in self.modalities.keys():
                    self.modalities['atac-g'] = {}
                
                self.modalities['atac-g'][sampkey] = data[sampkey]
                self.metadata.insert_row(prop)
        
        else: self.mudata.mod['atac.g'] = data

    def run_atac_call_peaks(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.atac_call_peaks, **kwargs)

    def run_atac_merge_peaks(self, run_on_samples = False, **kwargs):
        self.do_for_atac(run_on_samples, st.atac_merge_peaks, **kwargs)

    def run_atac_make_peak_matrix(self, run_on_samples = False, **kwargs):
        peak_matrix = self.do_for_atac(run_on_samples, st.atac_make_peak_matrix, **kwargs)
        if isinstance(peak_matrix, dict):
            for sampkey in peak_matrix.keys():
                prop = self.metadata.dataframe.loc[
                    (self.metadata.dataframe['sample'] == sampkey) &
                    (self.metadata.dataframe['modality'] == 'atac'), :
                ]

                assert len(prop) == 1
                prop = prop.iloc[0, :].copy()
                prop['modality'] = 'atac-p'
                if not 'atac-p' in self.modalities.keys():
                    self.modalities['atac-p'] = {}
                
                self.modalities['atac-p'][sampkey] = peak_matrix[sampkey]
                self.metadata.insert_row(prop)

    def run_atac_make_peak_matrix(self, run_on_samples = False, **kwargs):
        peak_matrix = self.do_for_atac(run_on_samples, st.atac_make_peak_matrix, **kwargs)
        if isinstance(peak_matrix, dict):
            for sampkey in peak_matrix.keys():
                prop = self.metadata.dataframe.loc[
                    (self.metadata.dataframe['sample'] == sampkey) &
                    (self.metadata.dataframe['modality'] == 'atac'), :
                ]

                assert len(prop) == 1
                prop = prop.iloc[0, :].copy()
                prop['modality'] = 'atac-p'
                if not 'atac-p' in self.modalities.keys():
                    self.modalities['atac-p'] = {}
                
                self.modalities['atac-p'][sampkey] = peak_matrix[sampkey]
                self.metadata.insert_row(prop)

    def run_atacp_filter_row_by_sum(self, run_on_samples = False, **kwargs):
        results = self.do_for_atac_peaks(run_on_samples, st.adata_filter_row_by_sum, **kwargs)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_atacp_filter_column_by_sum(self, run_on_samples = False, **kwargs):
        results = self.do_for_atac_peaks(run_on_samples, st.adata_filter_column_by_sum, **kwargs)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_atacp_annotate_peak(self, run_on_samples = False, **kwargs):
        self.do_for_atac_peaks(run_on_samples, st.atacp_annotate_peak, **kwargs)

    def run_atacp_markers_deseq(self, run_on_samples = False, **kwargs):
        self.do_for_atac_peaks(run_on_samples, st.rna_markers_deseq, **kwargs)

    def run_atacp_retrieve_sequence(self, run_on_samples = False, **kwargs):
        self.do_for_atac_peaks(run_on_samples, st.atacp_retrieve_sequence, **kwargs)

    def run_atacg_log_normalize(self, run_on_samples = False, **kwargs):
        self.do_for_atac_gene_activity(run_on_samples, st.rna_log_normalize, **kwargs)

    def run_atacg_filter_row_by_sum(self, run_on_samples = False, **kwargs):
        results = self.do_for_atac_gene_activity(run_on_samples, st.adata_filter_row_by_sum, **kwargs)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_atacg_filter_column_by_sum(self, run_on_samples = False, **kwargs):
        results = self.do_for_atac_gene_activity(run_on_samples, st.adata_filter_column_by_sum, **kwargs)
        if isinstance(results, dict):
            for k in results.keys():
                self.modalities['rna'][k] = results[k]
        else: self.mudata['rna'] = results

    def run_atacg_impute_magic(self, run_on_samples = False, **kwargs):
        self.do_for_atac_gene_activity(run_on_samples, st.rna_impute_magic, **kwargs) 

    def run_atacg_markers_deseq(self, run_on_samples = False, **kwargs):
        self.do_for_atac_gene_activity(run_on_samples, st.rna_markers_deseq, **kwargs)


    def run_rnaspc_qc(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_qc, **kwargs)
        
    def run_rnaspc_filter(self, run_on_samples = False):
        results = self.do_for_rnaspc(run_on_samples, st.rna_filter)
        self.modalities['rnasp-c'] = results

    def run_rnaspc_log_normalize(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_log_normalize, **kwargs)

    def run_rnaspc_select_hvg(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_select_hvg, **kwargs)

    def run_rnaspc_scale_pca(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_scale_pca, **kwargs)

    def run_rnaspc_knn(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_knn, **kwargs)

    def run_rnaspc_leiden(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_leiden, **kwargs)

    def run_rnaspc_leiden_subcluster(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_leiden_subcluster, **kwargs)

    def run_rnaspc_umap(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_umap, **kwargs)

    def run_rnaspc_mde(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_mde, **kwargs)

    def run_rnaspc_mde_fit(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_mde_fit, **kwargs)

    def run_rnaspc_transform(self, run_on_samples = False, xfunc = lambda x:x, yfunc = lambda x:x):
        self.do_for_rnaspc(run_on_samples, st.rnaspc_transform, xfunc = xfunc, yfunc = yfunc)

    def run_rnaspc_expression_mask(
        self, run_on_samples = False, gene = None, key = 'mask', 
        lognorm = 'X', threshold = 0.1, negate = False
    ):
        return self.do_for_rnaspc(
            run_on_samples, st.rna_expression_mask, 
            gene = gene, key = key, lognorm = lognorm, threshold = threshold,
            negate = negate
        )

    def run_rnaspc_integrate(self, method = 'harmony', dest = 'harmony', **kwargs):
        
        self.check_merged('rnasp-c')
        if method == 'harmony':
            from exprmat.preprocessing.integrate import harmony
            harmony(self.mudata['rnasp-c'], key = 'batch', adjusted_basis = dest, **kwargs)
        
        elif method == 'scanorama':
            from exprmat.preprocessing.integrate import scanorama
            scanorama(self.mudata['rnasp-c'], key = 'batch', adjusted_basis = dest, **kwargs)

        else: error(f'unsupported integration method `{method}`.')


    def run_rnaspc_markers(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_markers, **kwargs)

    def run_rnaspc_kde(self, run_on_samples = False, **kwargs):
        self.do_for_rnaspc(run_on_samples, st.rna_kde, **kwargs)

    def run_rnaspc_proportion(self, run_on_samples = False, **kwargs):
        '''
        This is a simplified method of cell type proportion calculation.
        It is implemented in earlier versions of the package and can be replaced by a more
        general version of counting summary. This returns a simple dataframe, while summary
        returns an annotated object and can be further processed using routines under
        ``exprmat.clustering.summary`` package.
        '''
        return self.do_for_rnaspc(run_on_samples, st.rna_proportion, **kwargs)

    def run_rnaspc_roi(self, run_on_samples = False, **kwargs):
        
        roi = self.do_for_rnaspc(run_on_samples, st.rnaspc_roi, **kwargs)
        if isinstance(roi, dict):
            roi = roi[kwargs.get('spsample')]
        
        roi.uns['spatial'] = {
            kwargs.get('sample_added'): roi.uns['spatial'][kwargs.get('spsample')]
        }

        rows = self.metadata.dataframe[(
            (self.metadata.dataframe['sample'] == kwargs.get('spsample')) &
            (self.metadata.dataframe['modality'] == 'rnasp-c')
        )]

        assert len(rows) == 1
        props = rows.iloc[0].copy()
        props['sample'] = kwargs.get('sample_added')
        props['modality'] = 'rnasp-c'

        self.metadata.insert_row(props)
        if not 'rnasp-c' in self.modalities.keys(): self.modalities['rnasp-c'] = {}
        self.modalities['rnasp-c'][kwargs.get('sample_added')] = roi
        self.modalities['rnasp-c'][kwargs.get('sample_added')].var = \
            st.search_genes(self.modalities['rnasp-c'][kwargs.get('sample_added')].var_names.tolist())



    # plotting wrappers

    def plot_rna_qc(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_qc, **kwargs)

    def plot_rna_embedding(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_embedding, **kwargs)
    
    def plot_rna_embedding_mask(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_embedding_mask, **kwargs)
    
    def plot_rna_embedding_atlas(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_embedding_atlas, **kwargs)

    def plot_rna_embedding_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_multiple_embedding, **kwargs)
    
    def plot_rna_embedding_atlas_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_multiple_embedding_atlas, **kwargs)

    def plot_rna_markers(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_markers, **kwargs)
    
    def plot_rna_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_dot, **kwargs)
    
    def plot_rna_heatmap(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_heatmap, **kwargs)
    
    def plot_rna_kde(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_kde, **kwargs)
    
    def plot_rna_proportion(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_proportion, **kwargs)
    
    def plot_rna_gene_gene(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_gene_gene, **kwargs)
    
    def plot_rna_gene_gene_regress(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_gene_gene_regress, **kwargs)

    def plot_rna_gene_gene_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_multiple_gene_gene, **kwargs)
    
    def plot_rna_cnv_matrix(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_cnv_matrix, **kwargs)
    
    def plot_rna_expression_bar(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_expression_bar, **kwargs)
    
    def plot_rna_expression_bar_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_expression_bar_multiple, **kwargs)
    
    def plot_rna_compare_scatter(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_compare_scatter, **kwargs)
    
    def plot_rna_qc_gene_counts(
        self, ncols = 4, figsize = (3, 3)
    ):
        from exprmat.utils import setup_styles
        setup_styles()

        from exprmat.preprocessing.plot import rna_plot_gene_histogram
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')
        if self.modalities is None: error('samples are not loaded')
        if not 'rna' in self.modalities.keys(): error('samples are not loaded')

        n_features = len(self.modalities['rna'])
        if n_features == 0: error('samples are not loaded')

        nrows = n_features // ncols
        if n_features % ncols != 0: nrows += 1
        fig, axes = plt.subplots(nrows, ncols)

        samples = list(self.modalities['rna'].keys())
        samples.sort()
        for feat_id in range(n_features):

            if len(axes.shape) == 2:
                rna_plot_gene_histogram(
                    self.modalities['rna'][samples[feat_id]],
                    sample_name = samples[feat_id],
                    ax = axes[feat_id // ncols, feat_id % ncols]
                )

            elif len(axes.shape) == 1:
                rna_plot_gene_histogram(
                    self.modalities['rna'][samples[feat_id]],
                    sample_name = samples[feat_id],
                    ax = axes[feat_id]
                )
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig
    
    def plot_rna_gsea_running_es(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_gsea_running_es, **kwargs)
    
    def plot_rna_gsea_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_gsea_dotplot, **kwargs)
    
    def plot_rna_opa_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_opa_dotplot, **kwargs)
    
    def plot_rna_lr_heatmap(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_lr_heatmap, **kwargs)
    
    def plot_rna_lr_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_lr_dotplot, **kwargs)
    
    def plot_rna_lr_circleplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_lr_circleplot, **kwargs)
    
    def plot_rna_volcano(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_volcano, **kwargs)
    
    def plot_rna_spliced_proportions(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_spliced_proportions, **kwargs)
    
    def plot_rna_velocity_gene(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_velocity_gene, **kwargs)
    
    def plot_rna_velocity_streamline(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_velocity_streamline, **kwargs)
    
    def plot_rna_cnmf_silhoutte(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_cnmf_silhoutte, **kwargs)
    
    def plot_rna_cnmf_density(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_cnmf_density, **kwargs)

    def plot_rna_cnmf_distance_comps(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_cnmf_distance_comps, **kwargs)

    def plot_rna_cnmf_distance_usages(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.rna_plot_cnmf_distance_usages, **kwargs)
    

    def plot_atac_qc(self, run_on_samples = False, **kwargs):
        return self.plot_for_atac(run_on_samples, st.atac_plot_qc, **kwargs)
    
    def plot_atac_embedding(self, run_on_samples = False, **kwargs):
        return self.plot_for_atac(run_on_samples, st.rna_plot_embedding, **kwargs)
    
    def plot_atacg_embedding(self, run_on_samples = False, **kwargs):
        return self.plot_for_atac_gene_activity(run_on_samples, st.rna_plot_embedding, **kwargs)
    

    def plot_rnaspc_qc(self, run_on_samples = False, **kwargs):
        return self.plot_for_rnaspc(run_on_samples, st.rna_plot_qc, **kwargs)

    def plot_rnaspc_embedding(self, run_on_samples = False, **kwargs):
        return self.plot_for_rnaspc(run_on_samples, st.rna_plot_embedding, **kwargs)
    
    def plot_rnaspc_embedding_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rnaspc(run_on_samples, st.rna_plot_multiple_embedding, **kwargs)
    
    def plot_rnaspc_embedding_spatial(self, run_on_samples = False, **kwargs):
        return self.plot_for_rnaspc(run_on_samples, st.rnaspc_plot_embedding_spatial, **kwargs)
    

    def plot_rnaspb_embedding_spatial(self, run_on_samples = False, **kwargs):
        return self.plot_for_rnaspb(run_on_samples, st.rnaspc_plot_embedding_spatial, **kwargs)
    

    def plot_sankey(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.adata_plot_sankey, **kwargs)
    
    def plot_matrix(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, st.adata_plot_matrix, **kwargs)
    


    # accessor wrappers

    def get_rna_markers(
        self, run_on_samples = False, de_slot = 'markers', group_name = None, max_q = None,
        min_pct = 0.25, max_pct_reference = 0.75, min_lfc = 1, max_lfc = 100, remove_zero_pval = False
    ):
        return self.do_for_rna(
            run_on_samples, st.rna_get_markers,
            de_slot = de_slot, group_name = group_name,
            max_q = max_q, min_pct = min_pct, min_lfc = min_lfc, max_lfc = max_lfc,
            max_pct_reference = max_pct_reference, remove_zero_pval = remove_zero_pval
        )
    
    def get_atacp_markers(
        self, run_on_samples = False, de_slot = 'markers', group_name = None, max_q = None,
        min_pct = 0.25, max_pct_reference = 0.75, min_lfc = 1, max_lfc = 100, remove_zero_pval = False
    ):
        return self.do_for_atac_peaks(
            run_on_samples, st.rna_get_markers,
            de_slot = de_slot, group_name = group_name,
            max_q = max_q, min_pct = min_pct, min_lfc = min_lfc, max_lfc = max_lfc,
            max_pct_reference = max_pct_reference, remove_zero_pval = remove_zero_pval
        )
    
    def get_atacg_markers(
        self, run_on_samples = False, de_slot = 'markers', group_name = None, max_q = None,
        min_pct = 0.25, max_pct_reference = 0.75, min_lfc = 1, max_lfc = 100, remove_zero_pval = False
    ):
        return self.do_for_atac_gene_activity(
            run_on_samples, st.rna_get_markers,
            de_slot = de_slot, group_name = group_name,
            max_q = max_q, min_pct = min_pct, min_lfc = min_lfc, max_lfc = max_lfc,
            max_pct_reference = max_pct_reference, remove_zero_pval = remove_zero_pval
        )

    def get_rna_lr(
        self, run_on_samples = False, lr_slot = 'lr', source_labels = None, target_labels = None,
        ligand_complex = None, receptor_complex = None, 
        filter_fun = None, top_n: int = None,
        orderby: str | None = None,
        orderby_ascending: bool | None = None,
        orderby_absolute: bool = False
    ):
        return self.do_for_rna(
            run_on_samples, st.rna_get_lr,
            lr_slot = lr_slot, source_labels = source_labels,
            target_labels = target_labels, ligand_complex = ligand_complex,
            receptor_complex = receptor_complex, filter_fun = filter_fun,
            top_n = top_n, orderby = orderby, orderby_ascending = orderby_ascending,
            orderby_absolute = orderby_absolute
        )
    
    def get_rna_gsea(
        self, run_on_samples = False, gsea_slot = 'gsea', max_fdr = 1.00, max_p = 0.05
    ):
        return self.do_for_rna(
            run_on_samples, st.rna_get_gsea, 
            gsea_slot = gsea_slot,
            max_fdr = max_fdr, max_p = max_p
        )
    
    def get_rna_opa(
        self, run_on_samples = False, opa_slot = 'opa', max_fdr = 1.00, max_p = 0.05
    ):
        return self.do_for_rna(
            run_on_samples, st.rna_get_opa, 
            gsea_slot = opa_slot, max_fdr = max_fdr, max_p = max_p
        )


    def save(self, fdir = None, save_samples = True):

        import os
        if fdir is None: fdir = self.directory

        os.makedirs(fdir, exist_ok = True)
        self.metadata.save(os.path.join(fdir, 'metadata.tsv'))

        if self.mudata is not None:
            if self.subset is None:
                info(f"main dataset write to {os.path.join(fdir, 'integrated.h5mu')}")
                self.mudata.write_h5mu(os.path.join(fdir, 'integrated.h5mu'))
            else: 
                info(f"main dataset write to {os.path.join(fdir, 'subsets', self.subset + '.h5mu')}")
                self.mudata.write_h5mu(os.path.join(fdir, 'subsets', self.subset + '.h5mu'))

        if not save_samples: return
        if self.modalities is not None:
            for key in self.modalities.keys():
                os.makedirs(os.path.join(fdir, key), exist_ok = True)
                for sample in self.modalities[key].keys():
                    
                    # save individual samples
                    self.modalities[key][sample].write_h5ad(
                        os.path.join(fdir, key, f'{sample}.h5ad')
                    )


    def push(
        self, 
        columns: list[str] | None = None, 
        mods: list[str] | None = None, 
        common: bool | None = None, 
        prefixed: bool | None = None, 
        drop: bool = False, 
        only_drop: bool = False
    ):
        self.mudata.push_obs(
            columns = columns, mods = mods, common = common,
            prefixed = prefixed, drop = drop, only_drop = only_drop
        )

    
    def pull(
        self,
        columns: list[str] | None = None,
        mods: list[str] | None = None, common: bool | None = None, 
        join_common: bool | None = None, 
        nonunique: bool | None = None, 
        join_nonunique: bool | None = None, 
        unique: bool | None = None, 
        prefix_unique: bool | None = True, 
        drop: bool = False, only_drop: bool = False
    ):
        self.mudata.pull_obs(
            columns = columns, mods = mods, 
            common = common, join_common = join_common,
            nonunique = nonunique, join_nonunique = join_nonunique,
            unique = unique, prefix_unique = prefix_unique,
            drop = drop, only_drop = only_drop
        )


    def joint(
        self, expm, name1 = 'self', name2 = 'addition',
        keep_obs = ['sample', 'batch', 'modality', 'taxa', 'group', 'cell.type'],
        keep_obsm = [],
        keep_layers = ['counts'],
        concat_label = 'origin',
        concat_dump = 'joint'
    ):
        
        concat_meta = metadata(
            locations = None, modality = None, default_taxa = None, 
            df = pd.concat(
                (self.metadata.dataframe, expm.metadata.dataframe),
                join = 'inner', ignore_index = True
        ))

        concat_dict = merge_dictionary(
            self.modalities,
            expm.modalities
        )

        self.pull(columns = keep_obs)
        expm.pull(columns = keep_obs)
        concat_mudata = mu.concat(
            {name1: self.mudata, name2: expm.mudata},
            join = 'outer', label = concat_label
        )

        merge = experiment(
            meta = concat_meta,
            mudata = concat_mudata,
            modalities = concat_dict,
            dump = concat_dump
        )

        if merge.mudata is not None:
            for mod in merge.mudata.mod_names:
                del merge.mudata[mod].obs

                for obsm in list(merge.mudata[mod].obsm_keys()):
                    if not obsm in keep_obsm:
                        del merge.mudata[mod].obsm[obsm]

                for layer in list(merge.mudata[mod].layers.keys()):
                    if not layer in keep_layers:
                        del merge.mudata[mod].layers[layer]
                

        merge.push(columns = keep_obs + ['origin'])
        merge.mudata['rna'].var = st.search_genes(concat_mudata['rna'].var.index)
        return merge
    

    # magic accessors

    def __getitem__(self, key):
        self.check_merged()
        if key in self.mudata.mod.keys():
            return self.mudata[key]
        else: 
            warning(f'key must be one of [{", ".join(list(self.mudata.mod.keys()))}]')
            error(f'no integrated modality named `{key}`.')
            

    def __repr__(self):

        from exprmat.ansi import green, cyan, red, yellow

        def print_anndata(adata: ad.AnnData):
            print(yellow('annotated data'), 'of size', adata.n_obs, '×', adata.n_vars)

            import textwrap
            if adata.obs is not None and len(adata.obs) > 0:
                print(green('    obs'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.obs.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])

            if adata.var is not None and len(adata.var) > 0:
                print(green('    var'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.var.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])
            
            if adata.layers is not None and len(adata.layers) > 0:
                print(green(' layers'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.layers.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])
            
            if adata.obsm is not None and len(adata.obsm) > 0:
                print(green('   obsm'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.obsm.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])
            
            if adata.varm is not None and len(adata.varm) > 0:
                print(green('   varm'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.varm.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])

            if adata.obsp is not None and len(adata.obsp) > 0:
                print(green('   obsp'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.obsp.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])

            if adata.varp is not None and len(adata.varp) > 0:
                print(green('   varp'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.varp.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])

            if adata.uns is not None and len(adata.uns) > 0:
                print(green('    uns'), ':', end = ' ')
                wrapped = textwrap.wrap(' '.join(adata.uns.keys()), width = 90)
                for nline in range(len(wrapped)):
                    if nline == 0: print(wrapped[nline])
                    else: print(' ' * 9, wrapped[nline])
            

        if self.mudata is not None:
            if self.subset is None:
                print(red('integrated dataset'), 'of size', self.mudata.n_obs, '×', self.mudata.n_vars)
            else: print(red('subset'), self.subset, 'of size', self.mudata.n_obs, '×', self.mudata.n_vars)
            print('contains modalities:', ', '.join([cyan(x) for x in list(self.mudata.mod.keys())]))

            for m in self.mudata.mod.keys():
                print('\n', 'modality', cyan(f'[{m}]'))
                print_anndata(self.mudata.mod[m])
            
            print()

        else: print(red('[!]'), 'dataset not integrated.')

        if self.modalities is None or len(self.modalities) == 0:
            print(red('[*]'), 'samples not loaded from disk.')
        
        else:
            print(red('[*]'), 'composed of samples:')
            for i_loc, i_sample, i_batch, i_grp, i_mod, i_taxa in zip(
                self.metadata.dataframe['location'], 
                self.metadata.dataframe['sample'], 
                self.metadata.dataframe['batch'], 
                self.metadata.dataframe['group'], 
                self.metadata.dataframe['modality'], 
                self.metadata.dataframe['taxa']
            ):
                if '.' in i_mod: continue
                loaded = False
                if (self.modalities is not None) and \
                   (i_mod in self.modalities.keys()) and \
                   (i_sample in self.modalities[i_mod].keys()):
                    loaded = True

                p_sample = i_sample if len(i_sample) < 30 else i_sample[:27] + ' ..'
                p_batch = i_batch if len(i_batch) < 30 else i_batch[:27] + ' ..'
                print(
                    f'  {p_sample:30}', cyan(f'{i_mod:4}'), yellow(f'{i_taxa:4}'),
                    f'batch {green(f"{p_batch:30}")}',
                    red('dataset not loaded') if not loaded else 
                    f'{green(str(self.modalities[i_mod][i_sample].n_obs))} × ' +
                    f'{yellow(str(self.modalities[i_mod][i_sample].n_vars))}'
                )

        return f'<exprmat.reader.experiment> ({len(self.metadata.dataframe)} samples)'

    pass


def merge_dictionary(dict1, dict2):

    for key in dict2.keys():
        if key not in dict1.keys():
            dict1[key] = dict2[key]
        elif isinstance(dict2[key], dict) and isinstance(dict1[key], dict):
            merge_dictionary(dict1[key], dict2[key])
        else: error('conflicting keys.')
    
    return dict1


def load_experiment(direc, load_samples = True, load_subset = None):
    
    import os
    if not os.path.exists(os.path.join(direc, 'metadata.tsv')):
        error('failed to load st. [metadata.tsv] file not found.')
    
    # read individual modality and sample
    meta = load_metadata(os.path.join(direc, 'metadata.tsv'))

    modalities = {}
    if load_samples:
        for modal, samp in zip(meta.dataframe['modality'], meta.dataframe['sample']):
            if '.' in modal: continue
            attempt = os.path.join(direc, modal, samp + '.h5ad')
            if os.path.exists(attempt):
                if modal not in modalities.keys(): modalities[modal] = {}
                modalities[modal][samp] = sc.read_h5ad(attempt)
            else: warning(f'sample dump [{modal}/{samp}] missing.')

    mdata = None
    subset = None
    if load_subset is None:
        if os.path.exists(os.path.join(direc, 'integrated.h5mu')):
            mdata = mu.read_h5mu(os.path.join(direc, 'integrated.h5mu'))
    else:
        subset = load_subset
        if os.path.exists(os.path.join(direc, 'subsets', load_subset + '.h5mu')):
            mdata = mu.read_h5mu(os.path.join(direc, 'subsets', load_subset + '.h5mu'))

    expr = experiment(
        meta = meta, 
        mudata = mdata, 
        modalities = modalities, 
        dump = direc,
        subset = subset
    )

    return expr
