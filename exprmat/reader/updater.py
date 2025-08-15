
import exprmat as em
from exprmat.ansi import error, warning, info

VERSION_CHECKPOINT = [1, em.SPECIFICATION]


def update(meta, modalities, mudata, version):

    # current version
    if version == em.SPECIFICATION: return version

    # valid version
    elif version in VERSION_CHECKPOINT:
        ckpt = VERSION_CHECKPOINT.index(version)
        for taskid in range(ckpt, len(VERSION_CHECKPOINT) - 1):
            task = UPDATE_TASK[taskid]
            info(f'updating experiment dump from {version} to {VERSION_CHECKPOINT[taskid + 1]} ...')
            task(meta, modalities, mudata)

        return VERSION_CHECKPOINT[taskid + 1]
    
    # invalid version
    else: error(f'illegal version specification of experiment dump: [{version}]')


def do_for(modalities, mudata, modality, func, **params):

    if modalities:
        if modality in modalities.keys():
            for k in modalities[modality].keys():
                func(modalities[modality][k], **params)
    if mudata:
        if modality in mudata.mod.keys():
            func(mudata.mod[modality], **params)

    
def update_1_39(meta, modalities, mudata):

    def update_rna(data):

        if 'ensembl' in data.var.columns:
            import pandas as pd
            obsn = data.var[['ensembl']].copy()
            taxa = obsn.index.tolist()[0].split(':')[1]
            from exprmat.data.finders import get_genome
            gtable = get_genome(taxa)
            gtable.index = gtable['id'].tolist()
            obsn = obsn.join(gtable, on = 'ensembl', how = 'left')
            assert len(obsn) == data.n_vars
            del obsn['ensembl']

            old_columns = [
                '.seqid', '.source', '.type', '.start', '.end', '.score', '.strand', '.phase',
                'biotype', 'source', 'gene', 'description', 'ensembl', 'uid', 'version', 'len.mean',
                'len.median', 'len.cds', '.ucsc'
            ]

            for o in old_columns:
                if o in data.var.columns: del data.var[o]
            data.var = pd.concat([obsn, data.var], axis = 1)

    def update_atac(data):
        data.var.rename(columns = {
            '.seqid': 'chr',
            '.start': 'start',
            '.end': 'end',
        }, inplace = True)

    do_for(modalities, mudata, 'rna', update_rna)
    do_for(modalities, mudata, 'rnasp-c', update_rna)
    do_for(modalities, mudata, 'rnasp-b', update_rna)
    do_for(modalities, mudata, 'rnasp-s', update_rna)
    do_for(modalities, mudata, 'atac-g', update_rna)

    do_for(modalities, mudata, 'atac', update_atac)
    do_for(modalities, mudata, 'atac-p', update_atac)
    pass


UPDATE_TASK = [
    update_1_39
]