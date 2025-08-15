
import os
import pandas as pd
import pickle
import numpy as np
import lightmotif

from exprmat.data.finders import basepath
from exprmat.ansi import error, warning, info

with open(os.path.join(basepath, 'shared', 'motifs.pkl'), 'rb') as f:
    motif_tree = pickle.load(f)

pssm_table = {}
transcription_factor_bindings = {}
motif_subsets = {}


def get_motifs(subset = 'all'):
    if subset == 'all': return motif_tree
    elif subset in motif_subsets.keys(): return motif_subsets[subset]
    else:
        subtree = {}
        for key in motif_tree.keys():
            items = {}
            for mot in motif_tree[key].keys():
                if motif_tree[key][mot]['source'] == subset:
                    items[mot] = motif_tree[key][mot]
            subtree[key] = items
        motif_subsets[subset] = subtree
        return motif_subsets[subset]


def get_transcription_factor_binding_motifs(taxa):

    if not taxa in transcription_factor_bindings.keys():
        transcription_factor_bindings[taxa] = \
            pd.read_feather(os.path.join(basepath, taxa, 'tf', 'transcription-factors.feather'))
    
    return transcription_factor_bindings[taxa]


def search_transcription_factor(taxa, factor, subset = 'all'):

    tfbindings = get_transcription_factor_binding_motifs(taxa)
    if factor in tfbindings['symbol'].tolist():
        row = tfbindings['symbol'].tolist().index(factor)
        row = tfbindings.iloc[row, :].copy()
        metaclust = row['metacluster'].split(' ')
        motifs = row['motif'].split(' ')

        table = get_motifs(subset)
        pairs = []
        for mc in metaclust:
            for mot in motifs:
                if mot in table[mc].keys():
                    pairs.append((mc, mot))
        
        return pairs
    
    else: error(f'could not find transcription factor {factor} in taxa {taxa}.')


def plot_sequence_logo(motif, fig_height = 1, fig_width_per_base = 0.2, dpi = 100):

    from exprmat.plotting.sequence import sequence_logo
    motifs = get_motifs()
    if isinstance(motif, tuple):
        return sequence_logo(
            np.array(motifs[motif[0]][motif[1]]['ppm'].T, dtype = np.float32), 
            ppm = True, y_format = 'bits', dpi = dpi, 
            fig_height = fig_height, fig_width_per_base = fig_width_per_base
        )
    
    elif isinstance(motif, str):
        for mc in motifs.keys():
            if motif in motifs[mc].keys():
                return sequence_logo(
                    np.array(motifs[mc][motif]['ppm'].T, dtype = np.float32), 
                    ppm = True, y_format = 'bits', dpi = dpi, 
                    fig_height = fig_height, fig_width_per_base = fig_width_per_base
                )
            

def motif_to_tuple(motif):
    motifs = get_motifs()
    for mc in motifs.keys():
        if motif in motifs[mc].keys():
            return (mc, motif)
    error(f'could not find motif {motif}')


def to_pssm(motif, eps = 0.01):
    
    if isinstance(motif, str): motif = motif_to_tuple(motif)
    if motif[1] in pssm_table.keys(): return pssm_table[motif[1]]

    motifs = get_motifs()
    integral = np.array(motifs[motif[0]][motif[1]]['ppm'].T * 100000, dtype = np.int32)
    
    # original order: acgt.
    pwm = lightmotif.CountMatrix({
        'A': integral[:, 0].tolist(),
        'C': integral[:, 1].tolist(),
        'T': integral[:, 3].tolist(),
        'G': integral[:, 2].tolist(),
    }).normalize(eps)

    pssm = pwm.log_odds()
    pssm_table[motif[1]] = pssm
    return pssm