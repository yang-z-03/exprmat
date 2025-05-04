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

from exprmat.reader.metadata import metadata, load_metadata
from exprmat.data.finders import get_genome
from exprmat.reader.matcher import read_mtx_rna
from exprmat.ansi import warning, info, error

class experiment:
    
    def __init__(self, meta : metadata, mudata = None, modalities = {}):

        # TODO: we support rna only at present.
        table = meta.dataframe.to_dict(orient = 'list')
        self.mudata = mudata
        self.modalities = modalities
        self.metadata = meta

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

            if i_mod == 'rna':
                if not 'rna' in self.modalities.keys(): self.modalities['rna'] = {}
                self.modalities['rna'][i_sample] = read_mtx_rna(
                    src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                    raw = False, default_taxa = i_taxa
                )
            
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
            
            self.variables['rna'] = pd.DataFrame(columns)
            self.variables['rna'].index = genes
        
        pass
    

    def merge(self, join = 'outer'):
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
                if 'qc' not in self.modalities['rna'][rnak].obs.columns: 
                    error(f'sample [{rnak}] have not been qc yet.')
                if 'qc' not in self.modalities['rna'][rnak].var.columns: 
                    error(f'sample [{rnak}] have not been qc yet.')
                
                filtered[rnak] = ad.AnnData(
                    self.modalities['rna'][rnak].X,
                    obs = self.modalities['rna'][rnak].obs,
                    var = self.modalities['rna'][rnak].var
                )[
                    self.modalities['rna'][rnak].obs['qc'],
                    self.modalities['rna'][rnak].var['qc']
                ].copy()

            # merge rna experiment.
            merged['rna'] = ad.concat(
                filtered, axis = 'obs', 
                join = join, label = 'sample'
            )

            # retrieve the corresponding gene info according to the universal 
            # nomenclature rna:[tax]:[ugene] format

            species_db = {}
            columns = {}
            gene_names = merged['rna'].var_names.tolist()

            n_genes = 0
            for gene in gene_names:
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
            
            for metakey in columns.keys():
                assert len(columns[metakey]) == merged['rna'].n_vars
                merged['rna'].var[metakey] = columns[metakey]
        
        if len(merged) > 0:
            mdata = mu.MuData(merged)
            mdata.push_obs()
            mdata.push_var()
            self.mudata = mdata

        else: self.mudata = None


    def do_for(self, samples = None, func, **kwargs):
        
        for mod, samp in zip(
            self.metadata.dataframe['modality'].tolist(),
            self.metadata.dataframe['sample'].tolist()
        ):
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

                func(self.modalities[mod][samp], **kwargs)


    def do_for_rna(self, func, **kwargs):
        self.do_for(self.all_rna_samples(), func, **kwargs)

    
    def all_samples(self):
        return self.metadata.dataframe['sample'].tolist()
    

    def all_rna_samples(self):
        return self.metadata.dataframe[
            self.metadata.dataframe['modality'] == 'rna', :
        ]['sample'].tolist()


    def rna_log_normalize(self):
        assert 'rna' in self.mudata.mod.keys()
        from exprmat.preprocessing import log_transform, normalize
        normalize(self)


    def save(self, fdir):

        import os

        os.makedirs(fdir, exist_ok = True)
        self.metadata.save(os.path.join(fdir, 'metadata.tsv'))

        if self.mudata is not None:
            self.mudata.write_h5mu(os.path.join(fdir, 'integrated.h5mu'))
        
        if self.modalities is not None:
            for key in self.modalities.keys():
                os.makedirs(os.path.join(fdir, key), exist_ok = True)
                for sample in self.modalities[key].keys():
                    
                    # save individual samples
                    self.modalities[key][sample].write_h5ad(
                        os.path.join(fdir, key, f'{sample}.h5ad')
                    )
    
    pass


def load_experiment(direc, load_samples = True):
    
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
    if os.path.exists(os.path.join(direc, 'integrated.h5mu')):
        mdata = mu.read_h5mu(os.path.join(direc, 'integrated.h5mu'))

    expr = experiment(meta = meta, mudata = mdata, modalities = modalities)
    expr.build_variables()
    return expr


class time_series_experiment(experiment):

    def __init__(self, meta : metadata, time_series_key):
        super().__init__(meta)
        self.key_time_series = time_series_key
        pass

    pass

