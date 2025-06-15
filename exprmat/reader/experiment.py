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
            
            info(f'reading sample {i_sample} [{i_mod}] ...')

            attempt_path = os.path.join(dump, i_mod, i_sample + '.h5ad')
            if os.path.exists(attempt_path):
                if not i_loc in self.modalities.keys(): self.modalities[i_mod] = {}
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
                    experiment.search_genes(self.modalities['rna'][i_sample].var_names.tolist())
                
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
                    
            
            else: warning(f'sample {i_sample} have no supported modalities')

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
    
    
    @staticmethod
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


            # merge rna experiment.
            merged['rna'] = ad.concat(
                filtered, axis = 'obs', 
                join = join, label = 'sample'
            )

            # retrieve the corresponding gene info according to the universal 
            # nomenclature rna:[tax]:[ugene] format

            gene_names = merged['rna'].var_names.tolist()
            merged['rna'].var = experiment.search_genes(gene_names)

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


        if len(merged) > 0:
            mdata = mu.MuData(merged)
            self.mudata = mdata

        else: self.mudata = None


    def do_for(self, samples, func, **kwargs):
        
        results = {}
        for mod, samp in zip(
            self.metadata.dataframe['modality'].tolist(),
            self.metadata.dataframe['sample'].tolist()
        ):
            if '.' in mod: continue

            results[samp] = None
            do = False
            if samples is None: do = True
            if isinstance(samples, list):
                if samp in samples: do = True

            if do:
                
                if self.modalities is None:
                    warning(f'experiment do not load any samples.')
                    return
                
                if not mod in self.modalities.keys():
                    warning(f'{mod} does not loaded in the modalities key.')
                    continue
                if not samp in self.modalities[mod].keys():
                    warning(f'{samp} not loaded in the {mod} modality.')
                    continue
                
                try:
                    results[samp] = func(self.modalities[mod][samp], samp, **kwargs)
                except: warning(f'method failed for sample {samp}')
        
        return results


    def do_for_rna(self, run_on_samples, func, **kwargs):
        if isinstance(run_on_samples, bool) and run_on_samples:
            return self.do_for(self.all_rna_samples(), func, **kwargs)
        elif isinstance(run_on_samples, list):
            return self.do_for(list(set(self.all_rna_samples()) & set(run_on_samples)), func, **kwargs)
        else:
            assert 'rna' in self.mudata.mod.keys()
            return func(self.mudata['rna'], 'integrated', **kwargs)
        
    
    def plot_for_rna(
        self, run_on_samples, func,
        run_on_splits = False, split_key = None, split_selection = None, **kwargs
    ):
        from exprmat.utils import setup_styles, plotting_styles
        setup_styles(**plotting_styles)

        if isinstance(run_on_samples, bool) and run_on_samples:
            figures = self.do_for(self.all_rna_samples(), func, **kwargs)
            for f in figures.values(): f.tight_layout()
            return figures
        
        elif isinstance(run_on_samples, list):
            figures = self.do_for(list(set(self.all_rna_samples()) & set(run_on_samples)), func, **kwargs)
            for f in figures.values(): f.tight_layout()
            return figures
        
        else:
            assert 'rna' in self.mudata.mod.keys()

            if not run_on_splits:
                figure = func(self.mudata['rna'], 'integrated', **kwargs)
                import warnings
                warnings.filterwarnings('ignore')
                figure.tight_layout()
                return figure

            else:
                results = {}
                cats = self.mudata['rna'].obs[split_key].cat.categories.tolist()
                if split_selection is None: split_selection = cats

                n_features = len(split_selection)
                for feat_id in range(len(split_selection)):

                    results[split_selection[feat_id]] = func(
                        self.mudata['rna'][
                            self.mudata['rna'].obs[split_key] == split_selection[feat_id],:].copy(),
                        # copy the data to silence the implicit modification warning on views. 
                        split_selection[feat_id], 
                        **kwargs
                    )

                    results[split_selection[feat_id]].tight_layout()
                
                return results

    
    def all_samples(self):
        return self.metadata.dataframe['sample'].tolist()
    

    def all_rna_samples(self):
        return self.metadata.dataframe[
            (self.metadata.dataframe['modality'] == 'rna')
        ]['sample'].tolist()
    

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


    @staticmethod
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


    @staticmethod
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

    
    @staticmethod
    def rna_filter(adata, sample_name):
        qc_cells = adata[adata.obs['qc'], adata.var['qc']].copy()
        # raw is rubbish. it does seldom over just throwing it.
        # qc_cells.raw = adata
        return qc_cells


    @staticmethod
    def rna_log_normalize(adata, sample_name, key_norm = 'norm', key_lognorm = 'lognorm', **kwargs):
        from exprmat.preprocessing import log_transform, normalize
        normalize(adata, counts = 'X', dest = key_norm, method = 'total')
        log_transform(adata, norm = key_norm, dest = key_lognorm)
        adata.layers['counts'] = adata.X
        adata.X = adata.layers[key_lognorm]

    
    @staticmethod
    def rna_select_hvg(adata, sample_name, key_lognorm = 'lognorm', method = 'vst', **kwargs):
        from exprmat.preprocessing import highly_variable
        highly_variable(
            adata, 
            counts = 'counts', lognorm = key_lognorm, 
            method = method, **kwargs
        )


    @staticmethod
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

    
    @staticmethod
    def rna_knn(adata, sample_name, **kwargs):
        from exprmat.reduction import run_knn
        run_knn(adata, **kwargs)


    @staticmethod
    def rna_leiden(adata, sample_name, **kwargs):
        from exprmat.clustering import run_leiden
        run_leiden(adata, **kwargs)


    @staticmethod
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


    @staticmethod
    def rna_umap(adata, sample_name, **kwargs):
        from exprmat.reduction import run_umap
        run_umap(adata, **kwargs)

    
    @staticmethod
    def rna_mde(adata, sample_name, data = 'pca', key_added = 'mde', **kwargs):
        from exprmat.reduction.mde import mde
        emb = mde(adata.obsm[data], **kwargs)
        adata.obsm[key_added] = emb

    
    @staticmethod
    def rna_mde_fit(
        adata, sample_name, data = 'pca', 
        based = 'umap', mask_key = 'sample', mask_values = [], 
        key_added = 'mde', **kwargs
    ):
        from exprmat.reduction.mde import mde_fit
        mask = [x in mask_values for x in adata.obs[mask_key]]
        emb = mde_fit(adata.obsm[data], fit = adata.obsm[based], fit_mask = mask, **kwargs)
        adata.obsm[key_added] = emb


    @staticmethod
    def rna_markers(adata, sample_name, **kwargs):
        from exprmat.descriptive.de import markers
        markers(adata, **kwargs)


    @staticmethod
    def rna_kde(adata, sample_name, **kwargs):
        from exprmat.descriptive.kde import density
        density(adata, **kwargs)

    
    @staticmethod
    def rna_proportion(
        adata, sample_name, major, minor, normalize = 'columns'
    ):
        if normalize == 'major': normalize = 'columns'
        if normalize == 'minor': normalize = 'index'
        tab = pd.crosstab(adata.obs[major], adata.obs[minor], normalize = normalize)
        return tab
    

    @staticmethod
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

    
    @staticmethod
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
    

    @staticmethod
    def rna_aggregate(
        adata, sample_name, data = 'X', method = 'mean', method_args = {},
        obs_key = 'sample', var_key = None
    ):
        from exprmat.descriptive.aggregate import aggregate
        return aggregate(
            adata, data = data, method = method, method_args = method_args,
            obs_key = obs_key, var_key = var_key
        )


    @staticmethod
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


    @staticmethod
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

    
    @staticmethod
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
    

    @staticmethod
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


    @staticmethod
    def rna_calculate_startracs_pairwise_metrics(
        adata, sample_name, base,
        clonotype = 'clone.id', cluster = 'leiden', key_added = 'tcr.cluster.ptrans'
    ):
        
        from exprmat.descriptive.tcr import (
            pairwise_transition
        )

        pairwise_transition(adata, base, clonotype = clonotype, cluster = cluster, key = key_added)


    @staticmethod
    def rna_expression_mask(adata, sample_name, gene, key, lognorm = 'X', threshold = 0.1, negate = False):
        from exprmat.utils import find_variable
        if not negate: adata.obs[key] = find_variable(adata, gene, layer = lognorm) >= threshold
        else: adata.obs[key] = find_variable(adata, gene, layer = lognorm) < threshold


    @staticmethod
    def rna_gsea(
        adata, sample_name, taxa,
        # differential expression slots:
        de_slot, group_name = None,
        min_pct = 0.0, max_pct_reference = 1, 
        min_lfc = None, max_lfc = None, remove_zero_pval = False,

        key_added = 'gsea',
        gene_sets = 'all',
        identifier = 'entrez'
    ):
        from exprmat.descriptive.gse import gse
        return gse(
            adata, taxa = taxa, de_slot = de_slot, group_name = None,
            min_pct = min_pct, max_pct_reference = max_pct_reference,
            min_lfc = min_lfc, max_lfc = max_lfc, remove_zero_pval = remove_zero_pval,
            key_added = key_added, gene_sets = gene_sets, identifier = identifier
        )
    

    @staticmethod
    def rna_opa(
        adata, sample_name, taxa,
        # differential expression slots:
        de_slot, group_name = None,
        min_pct = 0.0, max_pct_reference = 1, 
        min_lfc = None, max_lfc = None, remove_zero_pval = False,

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
            key_added = key_added, gene_sets = gene_sets, identifier = identifier,
            opa_cutoff = opa_cutoff, **kwargs
        )
    

    @staticmethod
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
    

    @staticmethod
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
    

    @staticmethod
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
    

    @staticmethod
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
                df = experiment.rna_gsva(
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

            df = experiment.rna_gsva(
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


    @staticmethod
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
    

    @staticmethod
    def rna_plot_qc(adata, sample_name, **kwargs):
        from exprmat.preprocessing.plot import rna_plot_qc_metrics
        return rna_plot_qc_metrics(adata, sample_name, **kwargs)
    

    @staticmethod
    def rna_plot_embedding(adata, sample_name, **kwargs):
        from exprmat.reduction.plot import embedding
        return embedding(adata, sample_name = sample_name, **kwargs)
    

    @staticmethod
    def rna_plot_embedding_atlas(adata, sample_name, **kwargs):
        from exprmat.reduction.plot import embedding_atlas
        return embedding_atlas(adata, sample_name = sample_name, **kwargs)
    

    @staticmethod
    def rna_plot_gene_gene(adata, sample_name, **kwargs):
        from exprmat.reduction.plot import gene_gene
        return gene_gene(adata, sample_name = sample_name, **kwargs)
    
    @staticmethod
    def rna_plot_gene_gene_regress(adata, sample_name, **kwargs):
        from exprmat.reduction.plot import gene_gene_regress
        return gene_gene_regress(adata, sample_name = sample_name, **kwargs)
    

    @staticmethod
    def rna_plot_markers(adata, sample_name, figsize, dpi, **kwargs):
        from exprmat.plotting.de import marker_plot
        pl = marker_plot(adata, sample_name = sample_name, **kwargs)
        pl.width = figsize[0]
        pl.height = figsize[1]
        pl.show()
        pl.fig.set_dpi(dpi)
        return pl.fig
    

    @staticmethod
    def rna_plot_expression_bar(
        adata, sample_name, gene, group, split = None,
        slot = 'X', selected_groups = None, palette = ['red', 'black'], 
        figsize = (6,3), dpi = 100
    ):
        from exprmat.plotting.expression import barplot
        pl = barplot(
            adata, gene = gene, slot = slot, group = group,
            split = split, selected_groups = selected_groups, palette = palette,
            size = figsize, dpi = dpi
        )
        return pl
    

    @staticmethod
    def rna_plot_expression_bar_multiple(
        adata, sample_name, features, ncols, group, split = None,
        slot = 'X', selected_groups = None, palette = ['red', 'black'], 
        figsize = (6,3), dpi = 100
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
                        size = figsize, dpi = dpi
                    )

                elif len(axes.shape) == 1:
                    barplot(
                        adata, gene = features[feat_id], slot = slot, group = group,
                        ax = axes[feat_id],
                        split = split, selected_groups = selected_groups, palette = palette,
                        size = figsize, dpi = dpi
                    )
            except: pass
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig
    

    @staticmethod
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
    

    @staticmethod
    def rna_plot_proportion(
        adata, sample_name, major, minor, plot = 'bar', cmap = 'Turbo',
        normalize = 'columns', figsize = (5,3), stacked = False, wedge = 0.4
    ):
        if normalize == 'major': normalize = 'columns'
        if normalize == 'minor': normalize = 'index'
        tmp = pd.crosstab(adata.obs[major], adata.obs[minor], normalize = normalize)

        def get_palette(n):
            if n + '.colors' in adata.uns.keys():
                return adata.uns[n + '.colors']
            else: return 'Turbo'

        if plot == 'bar':
            fig = tmp.plot.bar(stacked = stacked, figsize = figsize, grid = False)
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


    @staticmethod
    def rna_plot_kde(
        adata, sample_name, basis, kde, grouping_key, 
        figsize, dpi, groups = None, ncols = 1, **kwargs
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
                if len(axes.shape) == 2:
                    embedding(
                        adata[adata.obs[grouping_key] == groups[feat_id],:], 
                        basis, color = kde,
                        ax = axes[feat_id // ncols, feat_id % ncols], dpi = dpi,
                        sample_name = sample_name, title = groups[feat_id], **kwargs
                    )

                elif len(axes.shape) == 1:
                    embedding(
                        adata[adata.obs[grouping_key] == groups[feat_id],:], 
                        basis, color = kde,
                        ax = axes[feat_id], dpi = dpi,
                        sample_name = sample_name, title = groups[feat_id], **kwargs
                    )
            except: pass
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig
    

    @staticmethod
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
    

    @staticmethod
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
    

    @staticmethod
    def adata_plot_matrix(
        adata, sample_name, layer = 'X', obs_names = None, var_names = None,
        figsize = (3, 3), ax = None, **kwargs
    ):
        from exprmat.plotting.expression import matrix
        return matrix(
            adata, layer = layer, obs_names = obs_names, var_names = var_names,
            figsize = figsize, ax = ax, **kwargs
        )
    

    @staticmethod
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
    

    @staticmethod
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
    

    @staticmethod
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
    

    @staticmethod
    def rna_plot_cnv_matrix(adata, sample_name, **kwargs):
        from exprmat.plotting.cnv import chromosome_heatmap
        return chromosome_heatmap(adata, sample_name = sample_name, **kwargs)
    

    @staticmethod
    def rna_plot_gsea_running_es(
        adata, sample_name, gsea, terms, figsize = (4, 4), colors = None, title = None, **kwargs
    ):
        from exprmat.plotting.gse import esplot
        return esplot(
            adata, sample_name = sample_name, title = title,
            gsea = gsea, terms = terms, figsize = figsize, colors = colors, **kwargs
        )
    

    @staticmethod
    def rna_plot_gsea_dotplot(
        adata, sample_name, gsea_key, max_fdr = 1, max_p = 0.05, top_term: int = 100,
        colour = 'p', title = "", cmap = 'turbo', figsize = (3, 2), cutoff = 1, ptsize = 5,
        terms = None
    ):
        from exprmat.plotting.gse import gsea_dotplot
        return gsea_dotplot(
            experiment.rna_get_gsea(adata, gsea_key, max_fdr = max_fdr, max_p = max_p),
            column = colour, x = 'nes', y = 'name', title = gsea_key if title is None else title,
            cmap = cmap, size = ptsize, figsize = figsize, cutoff = cutoff, top_term = top_term,
            terms = terms
        )
    

    @staticmethod
    def rna_plot_opa_dotplot(
        adata, sample_name, opa_key, max_fdr = 1, max_p = 0.05, top_term: int = 100, terms = None,
        colour = 'fdr', title = None, cmap = 'turbo', figsize = (3, 2), cutoff = 1, ptsize = 5
    ):
        from exprmat.plotting.gse import opa_dotplot
        return opa_dotplot(
            experiment.rna_get_opa(adata, opa_key, max_fdr = max_fdr, max_p = max_p),
            column = colour, x = 'or', y = 'term', title = opa_key if title is None else title,
            cmap = cmap, size = ptsize, figsize = figsize, cutoff = cutoff, top_term = top_term,
            terms = terms
        )
    

    @staticmethod
    def rna_plot_lr_dotplot(adata, sample_name, lr_key, uns_key = None, **kwargs):
        from exprmat.plotting.lr import lr_dotplot
        return lr_dotplot(adata = adata, uns_key = lr_key, **kwargs)
    

    @staticmethod
    def rna_plot_lr_circleplot(adata, sample_name, lr_key, uns_key = None, **kwargs):
        from exprmat.plotting.lr import circleplot
        return circleplot(adata = adata, uns_key = lr_key, **kwargs)
    

    @staticmethod
    def rna_plot_lr_heatmap(adata, sample_name, lr_key, uns_key = None, **kwargs):
        from exprmat.plotting.lr import heatmap
        return heatmap(adata = adata, uns_key = lr_key, **kwargs)
    

    @staticmethod
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


    @staticmethod
    def rna_plot_spliced_proportions(adata, sample_name, **kwargs):
        from exprmat.plotting.velocity import proportions
        return proportions(adata, **kwargs)
    

    @staticmethod
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
    

    @staticmethod
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


    @staticmethod
    def adata_plot_sankey(adata, sample_name, obs1, obs2, exclude_values = ['na', 'nan'], **kwargs):
        from exprmat.plotting.sankey import sankey
        o1 = adata.obs[obs1].tolist()
        o2 = adata.obs[obs2].tolist()
        filters = [
            (ox1 not in exclude_values) and (ox2 not in exclude_values) 
            for ox1, ox2 in zip(o1, o2)
        ]

        return sankey(adata.obs.loc[filters, obs1], adata.obs.loc[filters, obs2], **kwargs)
    
    
    @staticmethod
    def rna_get_gsea(adata, gsea_slot = 'gsea', max_fdr = 1.00, max_p = 0.05):
        
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
    

    @staticmethod
    def rna_get_opa(adata, gsea_slot = 'gsea', max_fdr = 1.00, max_p = 0.05):
        
        df = pd.DataFrame(adata.uns[gsea_slot])

        if max_fdr is not None:
            df = df[df['fdr'] <= max_fdr]
        if max_p is not None:
            df = df[df['p'] <= max_p]
        
        df = df.sort_values(['fdr', 'p'])
        return df
    

    @staticmethod
    def rna_get_lr(
        adata, lr_slot = 'lr', source_labels = None, target_labels = None,
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


    @staticmethod
    def rna_get_markers(
        adata, de_slot = 'markers', group_name = None, max_q = None,
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


    # wrapper functions

    def run_rna_qc(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_qc, **kwargs)
        
    def run_rna_filter(self, run_on_samples = False):
        results = self.do_for_rna(run_on_samples, experiment.rna_filter)
        self.modalities['rna'] = results

    def run_rna_log_normalize(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_log_normalize, **kwargs)

    def run_rna_select_hvg(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_select_hvg, **kwargs)

    def run_rna_scale_pca(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_scale_pca, **kwargs)

    def run_rna_knn(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_knn, **kwargs)

    def run_rna_leiden(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_leiden, **kwargs)

    def run_rna_leiden_subcluster(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_leiden_subcluster, **kwargs)

    def run_rna_umap(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_umap, **kwargs)

    def run_rna_mde(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_mde, **kwargs)

    def run_rna_mde_fit(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_mde_fit, **kwargs)


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
        self.do_for_rna(run_on_samples, experiment.rna_markers, **kwargs)

    def run_rna_kde(self, run_on_samples = False, **kwargs):
        self.do_for_rna(run_on_samples, experiment.rna_kde, **kwargs)

    def run_rna_proportion(self, run_on_samples = False, **kwargs):
        '''
        This is a simplified method of cell type proportion calculation.
        It is implemented in earlier versions of the package and can be replaced by a more
        general version of counting summary. This returns a simple dataframe, while summary
        returns an annotated object and can be further processed using routines under
        ``exprmat.clustering.summary`` package.
        '''
        return self.do_for_rna(run_on_samples, experiment.rna_proportion, **kwargs)
    
    def run_rna_infercnv(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_infercnv, **kwargs)
    
    def run_rna_summary(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_summary, **kwargs)
    
    def run_rna_aggregate(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_aggregate, **kwargs)
    
    def run_rna_attach_tcr(self, run_on_samples = False):
        return self.do_for_rna(
            run_on_samples, 
            experiment.rna_attach_tcr, 
            searchdir = os.path.join(self.directory, 'tcr')
        )

    def run_rna_calculate_tcr_metrics(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_calculate_tcr_metrics, **kwargs)
    
    def run_rna_aggregate_tcr_by_identity(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_aggregate_tcr_by_identity, **kwargs)
    
    def run_rna_calculate_startracs_metrics(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_calculate_startracs_metrics, **kwargs)
    
    def run_rna_calculate_startracs_pairwise_metrics(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_calculate_startracs_pairwise_metrics, **kwargs)
    
    def run_rna_expression_mask(
        self, run_on_samples = False, gene = None, key = 'mask', 
        lognorm = 'X', threshold = 0.1, negate = False
    ):
        return self.do_for_rna(
            run_on_samples, experiment.rna_expression_mask, 
            gene = gene, key = key, lognorm = lognorm, threshold = threshold,
            negate = negate
        )
    
    def run_rna_gsea(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_gsea, **kwargs)
    
    def run_rna_opa(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_opa, **kwargs)
    
    def run_rna_gsva(self, run_on_samples = False, key_added = 'gsva', **kwargs):
        gsv = self.do_for_rna(run_on_samples, experiment.rna_gsva, **kwargs)
        if not run_on_samples:
            gsv.var['gset'] = gsv.var_names.tolist()
            gsv.var_names = [key_added + ':' + str(i + 1) for i in range(gsv.n_vars)]
            self.mudata.mod[key_added] = gsv
        
        return gsv
    
    def run_rna_remove_slots(self, run_on_samples = False, slot = 'obs', names = []):
        return self.do_for_rna(run_on_samples, experiment.remove_slot, slot = slot, names = names)
    
    def run_rna_ligand_receptor(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_ligand_receptor, **kwargs)
    
    def run_rna_score_genes(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_score_genes, **kwargs)
    
    def run_rna_score_genes_gsva(self, run_on_samples = False, key_added = 'gsva.scores', **kwargs):
        gsv = self.do_for_rna(run_on_samples, experiment.rna_score_genes_gsva, **kwargs)
        if not run_on_samples:
            gsv.var['gset'] = gsv.var_names.tolist()
            gsv.var_names = [key_added + ':' + str(i + 1) for i in range(gsv.n_vars)]
            self.mudata.mod[key_added] = gsv
        return gsv
    
    def run_rna_velocity(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_velocity, **kwargs)
    

    # plotting wrappers

    def plot_rna_qc(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_qc, **kwargs)

    def plot_rna_embedding(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_embedding, **kwargs)
    
    def plot_rna_embedding_atlas(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_embedding_atlas, **kwargs)

    def plot_rna_embedding_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_multiple_embedding, **kwargs)
    
    def plot_rna_embedding_atlas_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_multiple_embedding_atlas, **kwargs)

    def plot_rna_markers(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_markers, **kwargs)
    
    def plot_rna_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_dot, **kwargs)
    
    def plot_rna_heatmap(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_heatmap, **kwargs)
    
    def plot_rna_kde(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_kde, **kwargs)
    
    def plot_rna_proportion(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_proportion, **kwargs)
    
    def plot_rna_gene_gene(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_gene_gene, **kwargs)
    
    def plot_rna_gene_gene_regress(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_gene_gene_regress, **kwargs)

    def plot_rna_gene_gene_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_multiple_gene_gene, **kwargs)
    
    def plot_rna_cnv_matrix(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_cnv_matrix, **kwargs)
    
    def plot_rna_expression_bar(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_expression_bar, **kwargs)
    
    def plot_rna_expression_bar_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_expression_bar_multiple, **kwargs)
    
    def plot_rna_compare_scatter(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_compare_scatter, **kwargs)
    
    def plot_rna_qc_gene_counts(
        self, ncols = 4, figsize = (3, 3)
    ):
        from exprmat.utils import setup_styles, plotting_styles
        setup_styles(**plotting_styles)

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
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_gsea_running_es, **kwargs)
    
    def plot_rna_gsea_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_gsea_dotplot, **kwargs)
    
    def plot_rna_opa_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_opa_dotplot, **kwargs)
    
    def plot_rna_lr_heatmap(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_lr_heatmap, **kwargs)
    
    def plot_rna_lr_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_lr_dotplot, **kwargs)
    
    def plot_rna_lr_circleplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_lr_circleplot, **kwargs)
    
    def plot_rna_volcano(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_volcano, **kwargs)
    
    def plot_rna_spliced_proportions(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_spliced_proportions, **kwargs)
    
    def plot_rna_velocity_gene(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_velocity_gene, **kwargs)
    
    def plot_rna_velocity_streamline(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_velocity_streamline, **kwargs)
    
    def plot_sankey(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.adata_plot_sankey, **kwargs)
    
    def plot_matrix(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.adata_plot_matrix, **kwargs)
    

    # accessor wrappers

    def get_rna_markers(
        self, de_slot = 'markers', group_name = None, max_q = None,
        min_pct = 0.25, max_pct_reference = 0.75, min_lfc = 1, max_lfc = 100, remove_zero_pval = False
    ):
        self.check_merged('rna')
        return experiment.rna_get_markers(
            self.mudata['rna'], de_slot = de_slot, group_name = group_name,
            max_q = max_q, min_pct = min_pct, min_lfc = min_lfc, max_lfc = max_lfc,
            max_pct_reference = max_pct_reference, remove_zero_pval = remove_zero_pval
        )
    

    def get_rna_lr(
        self, lr_slot = 'lr', source_labels = None, target_labels = None,
        ligand_complex = None, receptor_complex = None, 
        filter_fun = None, top_n: int = None,
        orderby: str | None = None,
        orderby_ascending: bool | None = None,
        orderby_absolute: bool = False
    ):
        self.check_merged('rna')
        return experiment.rna_get_lr(
            self.mudata['rna'], lr_slot = lr_slot, source_labels = source_labels,
            target_labels = target_labels, ligand_complex = ligand_complex,
            receptor_complex = receptor_complex, filter_fun = filter_fun,
            top_n = top_n, orderby = orderby, orderby_ascending = orderby_ascending,
            orderby_absolute = orderby_absolute
        )
    

    def get_rna_gsea(
        self, gsea_slot = 'gsea', max_fdr = 1.00, max_p = 0.05
    ):
        self.check_merged('rna')
        return experiment.rna_get_gsea(
            self.mudata['rna'], gsea_slot = gsea_slot,
            max_fdr = max_fdr, max_p = max_p
        )
    

    def get_rna_opa(
        self, opa_slot = 'opa', max_fdr = 1.00, max_p = 0.05
    ):
        self.check_merged('rna')
        return experiment.rna_get_opa(
            self.mudata['rna'], gsea_slot = opa_slot,
            max_fdr = max_fdr, max_p = max_p
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
        merge.mudata['rna'].var = experiment.search_genes(concat_mudata['rna'].var.index)
        return merge
    

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
        error('failed to load experiment. [metadata.tsv] file not found.')
    
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
