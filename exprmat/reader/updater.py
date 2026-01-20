
import os
import h5py
from anndata.io import read_elem, write_elem

import exprmat as em
from exprmat.ansi import error, warning, info

VERSION_CHECKPOINT = [1, em.SPECIFICATION]


def update(meta, dump, version):

    # current version
    if version == em.SPECIFICATION: return version

    # valid version
    elif version in VERSION_CHECKPOINT:
        ckpt = VERSION_CHECKPOINT.index(version)
        for taskid in range(ckpt, len(VERSION_CHECKPOINT) - 1):
            task = UPDATE_TASK[taskid]
            info(f'updating experiment dump from {version} to {VERSION_CHECKPOINT[taskid + 1]} ...')
            
            # update samples
            for irow in range(len(meta.dataframe)):
                metadata = meta.dataframe.iloc[irow, :].copy()
                if '.' in metadata['modality']: continue
                info(f'updating sample ./{metadata["modality"]}/{metadata["sample"]} ...')
                task(metadata['modality'], os.path.join(dump, metadata["modality"], metadata["sample"] + '.h5ad'))
            
            # update subsets
            if os.path.exists(os.path.join(dump, 'subsets')):
                for subset in os.listdir(os.path.join(dump, 'subsets')):
                    if not subset.endswith('.h5mu'): continue
                    if subset.startswith('.'): continue
                    info(f'updating subset {subset.replace(".h5mu", "")} ...')
                    task('subset', os.path.join(dump, 'subsets', subset))
            
            if os.path.exists(os.path.join(dump, 'integrated.h5mu')):
                info(f'updating integrated dataset ...')
                task('integrated', os.path.join(dump, 'integrated.h5mu'))


        return VERSION_CHECKPOINT[taskid + 1]
    
    # invalid version
    else: error(f'illegal version specification of experiment dump: [{version}]')


def update_39(mod, fpath):

    def update_rna_var(table):
        if 'ensembl' in table.columns:
            import pandas as pd
            obsn = table[['ensembl']].copy()
            taxa = obsn.index.tolist()[0].split(':')[1]
            from exprmat.data.finders import get_genome
            gtable = get_genome(taxa)
            gtable.index = gtable['id'].tolist()
            obsn = obsn.join(gtable, on = 'ensembl', how = 'left')
            assert len(obsn) == len(table)
            del obsn['ensembl']

            old_columns = [
                '.seqid', '.source', '.type', '.start', '.end', '.score', '.strand', '.phase',
                'biotype', 'source', 'gene', 'description', 'ensembl', 'uid', 'version', 'len.mean',
                'len.median', 'len.cds', '.ucsc'
            ]

            for o in old_columns:
                if o in table.columns: del table[o]
            return pd.concat([obsn, table], axis = 1)

    def update_atac_var(table):
        return table.rename(columns = {
            '.seqid': 'chr',
            '.start': 'start',
            '.end': 'end',
        }, inplace = False)

    if not os.path.exists(fpath):
        warning(f'{fpath} is declared but do not exist on disk.')
        return
    
    file = h5py.File(fpath, 'r+')

    if mod in ['subset', 'integrated']:
        for m in list(file['mod'].keys()):
            if m in ['rna', 'rnasp-b', 'rnasp-c', 'rnasp-s', 'atac-g']:
                table = read_elem(file['mod'][m]['var'])
                table = update_rna_var(table)
                write_elem(file['mod'][m], 'var', table)
            
            elif m in ['atac', 'atac-p']:
                table = read_elem(file['mod'][m]['var'])
                table = update_atac_var(table)
                write_elem(file['mod'][m], 'var', table)
    
    if mod in ['rna', 'rnasp-b', 'rnasp-c', 'rnasp-s', 'atac-g']:
        table = read_elem(file['var'])
        table = update_rna_var(table)
        write_elem(file, 'var', table)
    
    elif mod in ['atac', 'atac-p']:
        table = read_elem(file['var'])
        table = update_atac_var(table)
        write_elem(file, 'var', table)
    
    file.close()


def trycatch(fun, *args):
    try:
        fun(*args)
    except:
        warning('routine failed and skipped.')

from functools import partial

UPDATE_TASK = [
    partial(trycatch, update_39)
]