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

from exprmat.reader.metadata import metadata, load_metadata
from exprmat.data.finders import get_genome
from exprmat.reader.matcher import read_mtx_rna
from exprmat.ansi import warning, info, error

class experiment:
    
    def __init__(self, meta : metadata, mudata = None):

        # TODO: we support rna only at present.
        table = meta.dataframe.to_dict(orient = 'list')
        self.mudata = mudata
        self.metadata = meta

        if self.mudata is not None:
            return

        modalities = {}
        for i_loc, i_sample, i_batch, i_grp, i_mod, i_taxa in zip(
            table['location'], table['sample'], table['batch'], table['group'],
            table['modality'], table['taxa']
        ):
            
            info(f'reading sample {i_sample} [{i_mod}] ...')

            if i_mod == 'rna':
                if not 'rna' in modalities.keys(): modalities['rna'] = {}
                modalities['rna'][i_sample] = read_mtx_rna(
                    src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                    raw = False, default_taxa = i_taxa
                )
            
            else: warning(f'sample {i_sample} have no supported modalities')

        # the var names are self-interpretable, and we will merge the samples
        # and throw away original column metadata. for atac-seq experiments, however,
        # the original var metadata is useful, we should store them and append
        # to the merged dataset later.

        if 'rna' in modalities.keys():

            # merge rna experiment.
            modalities['rna'] = ad.concat(
                modalities['rna'], axis = 'obs', 
                join = 'outer', label = 'sample'
            )

            # retrieve the corresponding gene info according to the universal 
            # nomenclature rna:[tax]:[ugene] format

            species_db = {}
            columns = {}
            gene_names = modalities['rna'].var_names.tolist()

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
                assert len(columns[metakey]) == modalities['rna'].n_vars
                modalities['rna'].var[metakey] = columns[metakey]
        
        if len(modalities) > 0:
            mdata = mu.MuData(modalities)
            mdata.push_obs()
            mdata.push_var()
            self.mudata = mdata
        else: self.mudata = None
                
        pass
    

    def save(self, fdir):

        import os
        if self.mudata is None:
            error('experiment is loaded with failure.')
        else: 
            self.mudata.write_h5mu(os.path.join(fdir, 'integrated.h5mu'))
            self.metadata.save(os.path.join(fdir, 'metadata.tsv'))
    
    pass


def load_experiment(direc):
    
    import os
    if not os.path.exists(os.path.join(direc, 'integrated.h5mu')):
        error('failed to load experiment. [integrated.h5mu] file not found.')
    if not os.path.exists(os.path.join(direc, 'metadata.tsv')):
        error('failed to load experiment. [metadata.tsv] file not found.')
    
    return experiment(
        meta = load_metadata(os.path.join(direc, 'metadata.tsv')),
        mudata = mu.read_h5mu(os.path.join(direc, 'integrated.h5mu'))
    )


class time_series_experiment(experiment):

    def __init__(self, meta : metadata, time_series_key):
        super().__init__(meta)
        self.key_time_series = time_series_key
        pass

    pass

