
import os
from exprmat.data.finders import basepath
from exprmat.data.cistarget import opendb

def get_tfs(taxa):

    with open(os.path.join(basepath, taxa, 'tf', 'transcription-factors.tsv')) as file:
        tfs_in_file = [line.strip() for line in file.readlines()]

    return tfs_in_file


def get_ranking_dbs(taxa, features = 'genes', cistromes = 'motifs'):

    dbs = os.listdir(os.path.join(
        basepath, taxa, 'tf', 'rankings', f'{features}-{cistromes}'
    ))

    dbs = [
        opendb(
            fname = os.path.join(basepath, taxa, 'tf', 'rankings', f'{features}-{cistromes}', fname),
            name = fname.replace('.feather', ''),
            expected_column_type = features,
            expected_row_type = cistromes,
            expected_score_or_ranking = 'rankings'
        ) for fname in dbs
    ]

    return dbs


def get_motif_annotation_fname(taxa):
    return os.path.join(
        basepath, taxa, 'tf', 'motif-to-tfs.tsv'
    )