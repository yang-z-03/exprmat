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
from exprmat.reader.matcher import read_mtx_rna
from exprmat.ansi import warning, info, error, red, green


class experiment:
    
    def __init__(self, meta : metadata, mudata = None, modalities = {}, dump = '.', subset = None):

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
                self.modalities['rna'][i_sample] = read_mtx_rna(
                    src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                    raw = False, default_taxa = i_taxa
                )
                self.modalities['rna'][i_sample].var = \
                    experiment.search_genes(self.modalities['rna'][i_sample].var_names.tolist())
            
            else: warning(f'sample {i_sample} have no supported modalities')

        # construct variable name table: note that this variable table represent the
        # all available set, and should be a superset of the merged table.
        
        self.build_variables()

        pass


    def build_variables(self):

        self.variables = {}

        if len(self.modalities) == 0 and self.mudata is not None:
            
            if 'rna' in self.mudata.mod.keys():
                self.variables['rna'] = self.mudata['rna'].var
            
            return

        if 'rna' in self.modalities.keys():
            genes = []
            for sample in self.modalities['rna'].keys():
                genes = set(list(genes) + self.modalities['rna'][sample].var_names.tolist())
            self.variables['rna'] = experiment.search_genes(genes)
        
        pass

    
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
        self, join = 'outer', variable_columns = [],
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

        if len(merged) > 0:
            mdata = mu.MuData(merged)
            mdata.push_obs()
            mdata.push_var()
            self.mudata = mdata

        else: self.mudata = None


    def do_for(self, samples, func, **kwargs):
        
        results = {}
        for mod, samp in zip(
            self.metadata.dataframe['modality'].tolist(),
            self.metadata.dataframe['sample'].tolist()
        ):
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

                results[samp] = func(self.modalities[mod][samp], samp, **kwargs)
        
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
        values = None, keys = 'sample'
    ):
        self.check_merged(slot)

        masks = []
        final = np.array([True] * self.mudata[slot].n_obs, dtype = np.bool)
        for idx in range(len(keys)):
            masks += [np.array([x in values[idx] for x in self.mudata[slot].obs[keys[idx]].tolist()])]
        
        for mask in masks: final = final & mask
        subset = self.mudata[final, :].copy()
        info(f'selected {final.sum()} observations from {len(final)}.')

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
        
    
    def exclude(
        self, slot = 'rna', annotation = 'cell.type', remove = []
    ):
        self.check_merged(slot)
        ls = self.mudata[slot].obs[annotation].tolist()
        mask = [x not in remove for x in ls]
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
    def rna_plot_qc(adata, sample_name, **kwargs):
        from exprmat.preprocessing.plot import rna_plot_qc_metrics
        return rna_plot_qc_metrics(adata, sample_name, **kwargs)
    

    @staticmethod
    def rna_plot_embedding(adata, sample_name, **kwargs):
        from exprmat.reduction.plot import embedding
        return embedding(adata, sample_name = sample_name, **kwargs)
    
    @staticmethod
    def rna_plot_gene_gene(adata, sample_name, **kwargs):
        from exprmat.reduction.plot import gene_gene
        return gene_gene(adata, sample_name = sample_name, **kwargs)
    

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

        if plot == 'bar':
            fig = tmp.plot.bar(stacked = stacked, figsize = figsize, grid = False)
            fig.legend(loc = None, bbox_to_anchor = (1, 1), frameon = False)
            fig.set_ylabel(f'Proportion ({minor})')
            fig.set_xlabel(major)
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
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig
    

    @staticmethod
    def rna_plot_dot(
        adata, sample_name, figsize, dpi, 
        var_names, groupby, *, log = False,
        categories_order = None, expression_cutoff = 0.0, 
        mean_only_expressed = False, standard_scale = 'var', 
        title = None, colorbar_title = 'Mean expression', 
        size_title = 'Fraction of cells (%)', gene_symbols = 'gene', 
        var_group_positions = None, var_group_labels = None, 
        var_group_rotation = None, layer = None, swap_axes = False, 
        dot_color_df = None, vmin = None, vmax = None, vcenter = None, norm = None, 
        cmap = 'turbo', dot_max = None, dot_min = None, smallest_dot = 0.0, **kwds
    ):
        from scanpy.plotting import dotplot
        pl = dotplot(
            adata, var_names, groupby = groupby, figsize = figsize,
            log = log, return_fig = True,
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
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig
    

    @staticmethod
    def rna_plot_cnv_matrix(adata, sample_name, **kwargs):
        from exprmat.plotting.cnv import chromosome_heatmap
        return chromosome_heatmap(adata, sample_name = sample_name, **kwargs)


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
        return self.do_for_rna(run_on_samples, experiment.rna_proportion, **kwargs)
    
    def run_rna_infercnv(self, run_on_samples = False, **kwargs):
        return self.do_for_rna(run_on_samples, experiment.rna_infercnv, **kwargs)

    
    # plotting wrappers

    def plot_rna_qc(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_qc, **kwargs)

    def plot_rna_embedding(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_embedding, **kwargs)

    def plot_rna_embedding_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_multiple_embedding, **kwargs)

    def plot_rna_markers(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_markers, **kwargs)
    
    def plot_rna_dotplot(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_dot, **kwargs)
    
    def plot_rna_kde(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_kde, **kwargs)
    
    def plot_rna_proportion(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_proportion, **kwargs)
    
    def plot_rna_gene_gene(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_gene_gene, **kwargs)

    def plot_rna_gene_gene_multiple(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_multiple_gene_gene, **kwargs)
    
    def plot_rna_cnv_matrix(self, run_on_samples = False, **kwargs):
        return self.plot_for_rna(run_on_samples, experiment.rna_plot_cnv_matrix, **kwargs)
    

    # accessor wrappers

    def get_rna_markers(
        self, de_slot = 'markers', group_name = None,
        min_pct = 0.25, max_pct_reference = 0.75, min_lfc = 1, max_lfc = 100, remove_zero_pval = False
    ):
        self.check_merged('rna')
        params = self.mudata['rna'].uns[de_slot]['params']

        # default value for convenience
        if len(self.mudata['rna'].uns[de_slot]['differential']) == 1 and group_name == None:
            group_name = list(self.mudata['rna'].uns[de_slot]['differential'].keys())[0]

        info('fetched diff `' + red(group_name) + '` over `' + green(params['reference']) + '`')
        tab = self.mudata['rna'].uns[de_slot]['differential'][group_name]

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
        
        tab = tab.sort_values(by = ['scores'], ascending = False)
        return tab


    def save(self, fdir = None, save_samples = True):

        import os
        if fdir is None: fdir = self.directory

        os.makedirs(fdir, exist_ok = True)
        self.metadata.save(os.path.join(fdir, 'metadata.tsv'))

        if self.mudata is not None:
            if self.subset is None:
                self.mudata.write_h5mu(os.path.join(fdir, 'integrated.h5mu'))
            else: self.mudata.write_h5mu(os.path.join(fdir, 'subsets', self.subset + '.h5mu'))

        if not save_samples: return
        if self.modalities is not None:
            for key in self.modalities.keys():
                os.makedirs(os.path.join(fdir, key), exist_ok = True)
                for sample in self.modalities[key].keys():
                    
                    # save individual samples
                    self.modalities[key][sample].write_h5ad(
                        os.path.join(fdir, key, f'{sample}.h5ad')
                    )
    

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

        print(red('[*]'), 'composed of samples:')
        for i_loc, i_sample, i_batch, i_grp, i_mod, i_taxa in zip(
            self.metadata.dataframe['location'], 
            self.metadata.dataframe['sample'], 
            self.metadata.dataframe['batch'], 
            self.metadata.dataframe['group'], 
            self.metadata.dataframe['modality'], 
            self.metadata.dataframe['taxa']
        ):
            loaded = False
            if (self.modalities is not None) and \
               (i_mod in self.modalities.keys()) and \
               (i_sample in self.modalities[i_mod].keys()):
                loaded = True

            print(
                f'  {i_sample:20}', cyan(f'{i_mod:4}'), yellow(f'{i_taxa:4}'),
                f'batch {green(f"{i_batch:10}")}',
                red('dataset not loaded') if not loaded else 
                f'{green(str(self.modalities[i_mod][i_sample].n_obs))} × ' +
                f'{yellow(str(self.modalities[i_mod][i_sample].n_vars))}'
            )

        return '<exprmat.reader.experiment>'

    pass


def load_experiment(direc, load_samples = True, load_subset = None):
    
    import os
    if not os.path.exists(os.path.join(direc, 'metadata.tsv')):
        error('failed to load experiment. [metadata.tsv] file not found.')
    
    # read individual modality and sample
    meta = load_metadata(os.path.join(direc, 'metadata.tsv'))

    modalities = {}
    if load_samples:
        for modal, samp in zip(meta.dataframe['modality'], meta.dataframe['sample']):
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

    expr.build_variables()
    return expr


class time_series_experiment(experiment):

    def __init__(self, meta : metadata, time_series_key):
        super().__init__(meta)
        self.key_time_series = time_series_key
        pass

    pass

