
import os
from exprmat.data.finders import basepath


def get_tfs(taxa):

    with open(os.path.join(basepath, taxa, 'tf', 'transcription-factors.tsv')) as file:
        tfs_in_file = [line.strip() for line in file.readlines()]

    return tfs_in_file