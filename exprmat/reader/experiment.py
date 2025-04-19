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

from exprmat.reader.metadata import metadata
from exprmat.reader.matcher import read_mtx_rna
from exprmat.ansi import warning, info

class experiment:
    
    def __init__(self, meta : metadata):

        # TODO: we support rna only at present.
        table = meta.dataframe.to_dict(orient = 'list')
        self.samples = {}

        for i_loc, i_sample, i_batch, i_grp, i_mod, i_taxa in zip(
            table['location'], table['sample'], table['batch'], table['group'],
            table['modality'], table['taxa']
        ):
            
            info(f'reading sample {i_sample} [{i_mod}] ...')
            modalities = {}
            if i_mod == 'rna':
                modalities['rna'] = read_mtx_rna(
                    src = i_loc, prefix = '', metadata = meta, sample = i_sample,
                    raw = False, default_taxa = i_taxa
                )

            if len(modalities) > 0:
                mdata = mu.MuData(modalities)
                mdata.push_obs()
                mdata.push_var()
                self.samples[i_sample] = mdata
                
            else: warning(f'sample {i_sample} have no supported modalities')
                
        pass

    pass


class time_series_experiment(experiment):

    def __init__(self, meta : metadata, time_series_key):
        super().__init__(meta)
        self.key_time_series = time_series_key
        pass

    pass

