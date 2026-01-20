
# Motif enrichment
# ----------------
#
# A hypergeometric test was used to test for overrepresentation of each DNA 
# motif in the set of differentially accessible peaks compared to a background 
# set of peaks. We tested motifs present in the JASPAR database [Fornes et al., 2020] 
# for human (species code 9606) by first identifying which peaks contained each 
# motif using the motifmatchr R package [Schep, 2020]. We computed the GC 
# content (percentage of G and C nucleotides) for each differentially accessible 
# peak and sampled a background set of 40,000 peaks such that the background set 
# was matched for overall GC content, accessibility, and peak width. This was 
# performed using the FindMotifs function in Signac.

import lightmotif
import os
import numpy as np
import scipy.sparse as sp
import scipy
from pynndescent import NNDescent

from exprmat.ansi import error, warning, info, pprog
from exprmat.data.finders import basepath
from exprmat import config as default


# genomic background sequencing
def fetch_background_sequences(gcbin, assembly, chr, n = 50000, length = 200, random_seed = 42):
    
    taxa = default['taxa.reference'][assembly]
    genome = os.path.join(basepath, taxa, 'assemblies', assembly, 'genome.fa.gz')
    from exprmat.data.finders import get_genome_size
    sizes = get_genome_size(assembly)

    import pyfastx
    import random
    random.seed(random_seed)
    fa = pyfastx.Fasta(genome)
    
    bg = { k: [] for k in gcbin }
    def check_complete(b, n):
        for v in b.values():
            if len(v) < n: return False
        return True
    
    nround = 1
    while not check_complete(bg, n):
        info(f'fetching chromosomes, round {nround} ...')
        # generate random indices for each chromosomes
        for c in chr:

            GEN_PER_CHR = 10000
            max_size = sizes[c] - length - 1
            random_intervals = [
                (x, x + length) for x in 
                [random.randint(1, max_size) for _ in range(GEN_PER_CHR)]
            ]

            rand_seq = fa.fetch(c, random_intervals, strand = '+')
            seqs = [rand_seq[x: x + length] for x in range(0, len(rand_seq), length)]
            gc_content = [
                ((x.count('C') + x.count('G')) / (len(x) - x.count('N'))) 
                if len(x) > x.count('N') else 0.50 for x in seqs
            ]
            gc_content = [(100 * x) // 5 for x in gc_content]
            gc_content = [19 if x == 20 else x for x in gc_content]

            for gc, seq in zip(gc_content, seqs):
                if (gc in gcbin) and (len(bg[gc]) < n): bg[gc].append(seq)
        
        nround += 1

    import pandas as pd
    pdf = pd.DataFrame(bg)
    pdf.columns = [str(x) for x in pdf.columns]
    return pdf


def gc_bias(adata, key_added = 'gc'):
    
    if ('strand.pos' not in adata.var.columns) or ('strand.neg' not in adata.var.columns):
        warning('could not find "strand.neg" and "strand.pos" in adata.var')
        error('you should run fetch_sequence(...) first on this atac-p dataset.')
    
    pos = adata.var['strand.pos'].tolist()
    pos = [x.upper() for x in pos]
    pos_gc = [x.count('C') + x.count('G') for x in pos]
    pos_len = [len(x) - x.count('N') for x in pos]
    gc = [x / y if y > 0 else 0.5 for x, y in zip(pos_gc, pos_len)]
    adata.var[key_added] = gc


def internal_background_peaks(
    adata, niterations = 50, n_jobs = -1, gc_key = 'gc', key_added = 'bg.intern'
):
    """
    Find background peaks based on GC bias and number of reads per peak

    Parameters
    ----------
    niterations : int, optional
        Number of background peaks to sample,, by default 50

    n_jobs : int, optional
        Number of cpus for compute. If set to -1, all cpus will be used, by default -1
    """

    # check if the object contains bias in Anndata.varm
    if not gc_key in adata.var.columns:
        error('cannot find .var["gc"], run `gc_bias(...)` first.')

    reads_per_peak = np.log1p(adata.X.sum(axis = 0)) / np.log(10)

    # here if reads_per_peak is a numpy matrix, convert it to array
    if sp.issparse(reads_per_peak): reads_per_peak = reads_per_peak.todense()
    if isinstance(reads_per_peak, np.matrix):
        reads_per_peak = np.squeeze(np.asarray(reads_per_peak))

    mat = np.array([reads_per_peak, adata.var[gc_key].values])
    chol_cov_mat = np.linalg.cholesky(np.cov(mat))
    trans_norm_mat = scipy.linalg.solve_triangular(
        a = chol_cov_mat, b = mat, lower = True).transpose()

    index = NNDescent(trans_norm_mat, metric = "euclidean",
                      n_neighbors = niterations, n_jobs = n_jobs)
    knn_idx, _ = index.query(trans_norm_mat, niterations)
    adata.varm[key_added] = knn_idx

    return None


def motif_enrichment(
    adata, background = 'background', n = 50000, length = 200, random_seed = 42,
    threshold = 1e-5, motif_matches = 'motifs', subset = 'jaspar', key_added = 'motifs.enrich'
):
    
    if ('strand.pos' not in adata.var.columns) or ('strand.neg' not in adata.var.columns):
        warning('could not find "strand.neg" and "strand.pos" in adata.var')
        warning(f'will fetch sequences from reference genome with central length {length}')
        from exprmat.peaks.sequence import query_sequence
        query_sequence(adata, length)

    pos = adata.var['strand.pos'].tolist()
    pos = [x.upper() for x in pos]
    revcomp = adata.var['strand.neg'].tolist()
    revcomp = [x.upper() for x in revcomp]

    # binning gc content
    pos_gc = np.array([x.count('C') + x.count('G') for x in pos])
    pos_len = np.array([len(x) - x.count('N') for x in pos])
    
    # find motifs in peaks:
    from exprmat.data.cre import get_motifs, to_pssm
    motifs = get_motifs(subset)
    # NOTE: DEBUG
    # motifs = {k: motifs[k] for k in list(motifs.keys())[0:10]}
    
    pssm = {}
    for mc in pprog(motifs.keys(), desc = 'generating pssm'):
        for mot in motifs[mc].keys():
            pssm[mot] = to_pssm((mc, mot))

    if (motif_matches in adata.varm.keys()) and \
        (f'{motif_matches}.names' in adata.uns.keys()) and \
        (f'{motif_matches}.seqlen' in adata.uns.keys()) and \
        adata.uns[f'{motif_matches}.names'] == list(pssm.keys()):
        mat = adata.varm[motif_matches].T
    else:
        do_match_motif(adata, motif_matches, subset, length, threshold)
        mat = adata.varm[motif_matches].T

    foreground = {}; foreground_gc_ratio = {} 
    # recover statistics from the precomputed matrix (motifs by peaks).
    for i, key in enumerate(list(pssm.keys())):
        matches = np.array(mat[i, :].todense())[0]
        foreground[key] = matches.sum()
        matches = matches == 1
        foreground_gc_ratio[key] = \
            (pos_gc[matches].sum() / pos_len[matches].sum()) \
                if pos_len[matches].sum() > 0 else 0.5
    
    adata.uns[f'{motif_matches}.names'] = list(pssm.keys())

    gcbin = [min(19, (100 * x) // 5) for x in foreground_gc_ratio.values()]
    gcbin = list(set(gcbin))
    gcbin = [int(x) for x in gcbin]

    info('generating background regions ...')
    if background in adata.uns.keys():
        bgseq = adata.uns[background]
    else:
        bgseq = fetch_background_sequences(
            gcbin, adata.uns['assembly'], 
            adata.var['chr'].unique().tolist(),
            n, length, random_seed
        )
        adata.uns[background] = bgseq

    # find motifs in background
    info('matching motifs for background ...')
    background_hit = {}
    background_len = {}
    for pkey in pprog(pssm.keys(), desc = 'background matching'):
        ind = int(min(19, (foreground_gc_ratio[pkey] * 100) // 5))
        # mark the central 200bp
        concat = ''
        for seq in bgseq[str(ind)]: concat += seq
        striped = lightmotif.stripe(concat)
        scores = pssm[pkey].calculate(striped)
        matches = scores.threshold(pssm[pkey].score(threshold))

        masked_indices = length - len(pssm[pkey])
        valid_matches = [x for x in matches if (x % length) < masked_indices]
        valid_segments = list(set([x // length for x in valid_matches]))

        background_hit[pkey] = len(valid_segments)
        background_len[pkey] = len(bgseq[str(ind)])

    # construct table
    key = []
    fg_hit = []
    fg_len = []
    fl = len(pos)
    gc_pct = []
    bg_hit = []
    bg_len = []
    for k in foreground.keys():
        fg_hit.append(foreground[k])
        fg_len.append(fl)
        gc_pct.append(foreground_gc_ratio[k])
        bg_hit.append(background_hit[k])
        bg_len.append(background_len[k])
        key.append(k)
    
    import pandas
    data = pandas.DataFrame({
        'motif': key,
        'fg.hit': fg_hit,
        'fg': fg_len,
        'gc': gc_pct,
        'bg.hit': bg_hit,
        'bg': bg_len
    })

    # filter for zero detection
    data = data.loc[data['fg.hit'] != 0, :].copy()
    data = data.loc[data['bg.hit'] != 0, :].copy()

    data['fg.ratio'] = data['fg.hit'] / data['fg']
    data['bg.ratio'] = data['bg.hit'] / data['bg']
    data['fc'] = data['fg.ratio'] / data['bg.ratio']

    # chisquare test
    n = (data['fg'] + data['bg']).to_numpy()
    b = (data['fg'] - data['fg.hit']).to_numpy() / n
    a = data['fg.hit'].to_numpy() / n
    c = data['bg.hit'].to_numpy() / n
    d = (data['bg'] - data['bg.hit']).to_numpy() / n
    z = (a * d - c * b)
    z = z * z
    y = (a + b) * (c + d) * (a + c) * (b + d)
    data['chisq'] = n * z / y

    import scipy.stats
    data['p'] = [
        1 - scipy.stats.chi2.cdf(abs(chi2_score), 1)
        for chi2_score in data['chisq']
    ]

    import warnings
    warnings.filterwarnings('ignore')
    data['log10.p'] = - np.log10(data['p'])
    warnings.filterwarnings('default')
    adata.uns[key_added] = data


def match_motif(length, threshold, pos, revcomp, pssm):
    
    info('matching motifs for foreground ...')
    match_matrix_rowind = []
    match_matrix_colind = []
    match_matrix_values = []
    loc_matrix_rowind = []
    loc_matrix_colind = []
    loc_matrix_values = []
    
    rowind = 0
    for pkey in pprog(pssm.keys(), desc = 'foreground matching'):
        concat = ''
        hlen = length // 2
        for seq in pos + revcomp: concat += seq[len(seq) // 2 - hlen : len(seq) // 2 + hlen]
        striped = lightmotif.stripe(concat)
        scores = pssm[pkey].calculate(striped)
        matches = scores.threshold(pssm[pkey].score(threshold))

        masked_indices = length - len(pssm[pkey])
        pssm_len = len(pssm[pkey])
        valid_matches = [x for x in matches if (x % length) < masked_indices]
        valid_segments = [x // length for x in valid_matches]
        
        segment_pos = [x % length for x in valid_matches]
        segment_pos = [
            (1 + x + pssm_len // 2) if (v < len(pos)) else -(length - x - pssm_len // 2)
            for x, v in zip(segment_pos, valid_segments)
        ]

        valid_segments = [x - len(pos) if x >= len(pos) else x for x in valid_segments]

        # remove duplicates
        valid_segments, ind = np.unique(valid_segments, return_index = True)
        segment_pos = np.array(segment_pos)[ind]

        # construct matrix.
        match_matrix_rowind += [rowind] * len(valid_segments)
        match_matrix_colind += valid_segments.tolist()
        match_matrix_values += [1] * len(valid_segments)
        loc_matrix_rowind += [rowind] * len(valid_segments)
        loc_matrix_colind += valid_segments.tolist()
        loc_matrix_values += segment_pos.tolist()
        rowind += 1

    mat = sp.csr_matrix(
        (match_matrix_values, (match_matrix_rowind, match_matrix_colind)), 
        shape = (rowind, len(pos)), dtype = np.uint8
    )

    loc = sp.csr_matrix(
        (loc_matrix_values, (loc_matrix_rowind, loc_matrix_colind)), 
        shape = (rowind, len(pos)), dtype = np.int32
    )

    return mat, loc


def do_match_motif(
    adata, motif_matches = 'motifs', subset = 'jaspar', 
    length = 200, threshold = 1e-5
):
    from exprmat.data.cre import get_motifs, to_pssm
    motifs = get_motifs(subset)
    # NOTE: DEBUG
    # motifs = {k: motifs[k] for k in list(motifs.keys())[0:10]}

    pssm = {}
    for mc in pprog(motifs.keys(), desc = 'generating pssm'):
        for mot in motifs[mc].keys():
            pssm[mot] = to_pssm((mc, mot))

    if ('strand.pos' not in adata.var.columns) or \
        ('strand.neg' not in adata.var.columns):
        warning('cannot find `strand.pos` and `strand.neg` in .var slots.')
        warning(f'will fetch sequences from reference genome with central length {length}')
        from exprmat.peaks.sequence import query_sequence
        query_sequence(adata, length)

    pos = adata.var['strand.pos'].tolist()
    pos = [x.upper() for x in pos]
    revcomp = adata.var['strand.neg'].tolist()
    revcomp = [x.upper() for x in revcomp]
    mat, loc = match_motif(length, threshold, pos, revcomp, pssm)
    adata.varm[motif_matches] = mat.T
    adata.varm[f'{motif_matches}.locs'] = loc.T
    adata.uns[f'{motif_matches}.names'] = list(pssm.keys())
    adata.uns[f'{motif_matches}.seqlen'] = length


def compute_deviations(
    adata, n_jobs = -1, chunk_size : int = 10000,
    background = 'bg.intern',
    motif_matches = 'motifs',
    gc_key = 'gc', subset = 'jaspar',
    length = 200, threshold = 1e-5,
    n_background = 50,
):
    
    if not gc_key in adata.var.keys():
        gc_bias(adata, key_added = gc_key)

    # check if the object contains bias in Anndata.varm
    if (not background in adata.varm.keys()):
        warning("cannot find background peaks in the input object .varm")
        warning(f"calculating internal background peaks and inserting to .varm['{background}']")
        internal_background_peaks(adata, key_added = background, niterations = n_background, gc_key = gc_key)
    
    if adata.varm[background].shape[1] != n_background:
        warning(f'detected existing background with size {adata.varm[background].shape}')
        warning(f'however, you requested that the `n_background` be {n_background}')
        warning(f'will re-generate the internal background with the number of iterations you specified.')
        internal_background_peaks(adata, key_added = background, niterations = n_background, gc_key = gc_key)

    info('computing expectation reads per cell and peak ...')
    expectation_obs, expectation_var = compute_expectation(count = adata.X)

    if not motif_matches in adata.varm.keys():
        info('querying motifs in peaks ...')
        do_match_motif(adata, motif_matches, subset, length, threshold)

    info('computing observed + bg motif deviations...')
    motif_match = adata.varm[motif_matches]
    obs_dev = np.zeros((adata.n_obs, motif_match.shape[1]), dtype = np.float32)

    # compute background deviations for bias-correction
    n_bg_peaks = adata.varm[background].shape[1]
    bg_dev = np.zeros(
        shape = (n_bg_peaks, adata.n_obs, len(
        adata.uns[f'{motif_matches}.names'])
    ), dtype = np.float32)
    
    # instead of iterating over bg peaks, iterate over X
    for item in adata.chunked_X(chunk_size):
        X, start, end = item
        info(f'calculating for rows (cells) {start} - {end} ...')
        obs_dev[start:end, :] = deviations((motif_match, X, expectation_obs[start:end], expectation_var))
        for i in pprog(range(n_bg_peaks), desc = "background iterations"):
            bg_peak_idx = adata.varm[background][:, i]
            bg_motif_match = adata.varm[motif_matches][bg_peak_idx, :]
            bg_dev[i, start:end, :] = deviations((bg_motif_match, X, expectation_obs[start:end], expectation_var))
    
    mean_bg_dev = np.mean(bg_dev, axis = 0)
    std_bg_dev = np.std(bg_dev, axis = 0)

    import warnings 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dev = (obs_dev - mean_bg_dev) / std_bg_dev
        dev = np.nan_to_num(dev, 0).astype('float32')

    import anndata as ad
    dev = ad.AnnData(dev)
    dev.obs_names = adata.obs_names
    dev.obs = adata.obs.copy()
    dev.var_names = adata.uns[f'{motif_matches}.names']
    return dev


def deviations(arguments):

    motif_match, count, expectation_obs, expectation_var = arguments
    # motif_match: n_var x n_motif
    # count, exp: n_obs x n_var

    # NOTE: motif_match must be converted to dense matrix!
    #       or else this will trigger mysterious bugs that eats up all the memory.
    observed = count.dot(motif_match)
    if sp.issparse(motif_match): motif_match = motif_match.todense()
    expected = expectation_obs.dot(expectation_var.dot(motif_match))
    if sp.issparse(observed):
        observed = observed.todense()
    if sp.issparse(expected):
        expected = expected.todense()
    out = np.zeros(expected.shape, dtype = expected.dtype)
    np.divide(observed - expected, expected, out = out, where=expected != 0)
    return out


def compute_expectation(count) -> np.array:
    """
    Compute expetation accessibility per peak and per cell by assuming
    identical read probability per peak for each cell with a sequencing
    depth matched to that cell observed sequencing depth

    Parameters
    ----------
    count : Union[np.array, sparse.csr_matrix]
        Count matrix containing raw accessibility data.

    Returns
    -------
    np.array, np.array
        Expectation matrix pair when multiplied gives
    """
    a = np.asarray(count.sum(0), dtype=np.float32).reshape((1, count.shape[1]))
    a /= a.sum()
    b = np.asarray(count.sum(1), dtype=np.float32).reshape((count.shape[0], 1))
    return b, a