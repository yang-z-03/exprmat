
import os
from pathlib import Path
import pandas as pd
import cython

from exprmat.snapatac import AnnData, AnnDataSet
import exprmat.snapatac as internal
from exprmat.ansi import info, warning, error


def subpeak_letters(i: cython.int) -> str:
    if i < 26: return chr(97+i)
    else: return subpeak_letters(i // 26) + chr(97 + (i % 26))
    

def peakio_to_pandas(peakio, score_column: str = "score"):

    import cython
    from itertools import groupby
    from operator import itemgetter

    n_peak: cython.int
    chrom: bytes
    s: cython.long
    peakname: str

    chrs = list(peakio.peaks.keys())
    n_peak = 0

    columns = {
        'chr': [],
        'start': [],
        'end': [],
        'peak': [],
        'score': [],
        'strand': [],
        'fc': [],
        'p': [],
        'q': [],
        'summit': []
    }

    for chrom in sorted(chrs):
        for end, group in groupby(peakio.peaks[chrom], key = itemgetter("end")):
            n_peak += 1
            these_peaks = list(group)
            
            if len(these_peaks) > 1:  # from call-summits
                for i, peak in enumerate(these_peaks):
                    peakname = "peak:%d%s" % (n_peak, subpeak_letters(i))
                    if peak['summit'] == -1: s = -1
                    else: s = peak['summit'] - peak['start']

                    columns['chr'].append(chrom.decode())
                    columns['start'].append(peak['start'])
                    columns['end'].append(peak['end'])
                    columns['peak'].append(peakname)
                    columns['score'].append(int(10 * peak[score_column]))
                    columns['fc'].append(peak['fc'])
                    columns['p'].append(peak['pscore'])
                    columns['q'].append(peak['qscore'])
                    columns['summit'].append(s)
                    columns['strand'].append('.')

            else:
                peak = these_peaks[0]
                peakname = "peak:%d" % (n_peak)
                if peak['summit'] == -1: s = -1
                else: s = peak['summit'] - peak['start']

                columns['chr'].append(chrom.decode())
                columns['start'].append(peak['start'])
                columns['end'].append(peak['end'])
                columns['peak'].append(peakname)
                columns['score'].append(int(10 * peak[score_column]))
                columns['fc'].append(peak['fc'])
                columns['p'].append(peak['pscore'])
                columns['q'].append(peak['qscore'])
                columns['summit'].append(s)
                columns['strand'].append('.')
    
    return pd.DataFrame(columns)


def call_peak_from_bedgraph(
    bedgraph_file,
    cutoff = 1, minlen = 200, maxgap = 30, call_summits = False
):

    from MACS3.IO import BedGraphIO
    from MACS3.Utilities.Logger import logging

    import logging
    logging.getLogger().setLevel(logging.CRITICAL + 1) # temporarily disable logging

    info("reading and building bedgraph ...")
    bio = BedGraphIO.bedGraphIO(bedgraph_file)
    btrack = bio.read_bedGraph(baseline_value = 0)

    info("calling peaks ...")
    peaks = btrack.call_peaks(
        cutoff = float(cutoff), 
        min_length = int(minlen), 
        max_gap = int(maxgap), 
        call_summits = call_summits
    )

    return peakio_to_pandas(peaks)


def call_peak_from_fragments(
    adata: AnnData | AnnDataSet,
    *,
    groupby: str | list[str] | None = None,
    qvalue: float = 0.05,
    call_broad_peaks: bool = False,
    broad_cutoff: float = 0.1,
    replicate: str | list[str] | None = None,
    replicate_qvalue: float | None = None,
    max_frag_size: int | None = 180,
    selections: set[str] | None = None,
    nolambda: bool = False,
    shift: int = -100,
    extsize: int = 200,
    min_len: int | None = None,
    blacklist: Path | None = None,
    key_added: str = 'macs3',
    tempdir: Path | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
):
    """
    Call peaks using MACS3 from single-cell fragments.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.

    groupby
        Group the cells before peak calling. If a `str`, groups are obtained from
        `.obs[groupby]`. If None, peaks will be called for all cells.

    qvalue
        qvalue cutoff used in MACS3.

    call_broad_peaks
        If True, MACS3 will call broad peaks. The broad peak calling process
        utilizes two distinct cutoffs to discern broader, weaker peaks (`broad_cutoff`)
        and narrower, stronger peaks (`qvalue`), which are subsequently nested to
        provide a detailed peak landscape. To conceptualize "nested" peaks, picture
        a gene structure housing regions analogous to exons (strong peaks) and
        introns coupled with UTRs (weak peaks). Please note that, if you only want to
        call "broader" peak and not interested in the nested peak structure, please
        simply use `qvalue` with weaker cutoff instead of using `call_broad_peaks` option.

    broad_cutoff
        qvalue cutoff used in MACS3 for calling broad peaks.

    replicate
        Replicate information. If provided, reproducible peaks will be called for each group.

    replicate_qvalue
        qvalue cutoff used in MACS3 for calling peaks in replicates.
        This parameter is only used when `replicate` is provided. Typically this parameter 
        is used to call peaks in replicates with a more lenient cutoff. If not provided, 
        `qvalue` will be used.

    max_frag_size
        Maximum fragment size. If provided, fragments with sizes larger than `max_frag_size` 
        will be not be used in peak calling. This is used in ATAC-seq data to remove fragments 
        that are not from nucleosome-free regions. You can check the fragment size distribution
        to choose a proper value for this parameter.

    selections
        Call peaks for the selected groups only.

    nolambda
        If True, macs3 will use the background lambda as local lambda.
        This means macs3 will not consider the local bias at peak candidate regions.

    shift
        The shift size in MACS.

    extsize
        The extension size in MACS.

    min_len
        The minimum length of a called peak. If None, it is set to `extsize`.

    blacklist
        Path to the blacklist file in BED format. If provided, regions in the blacklist 
        will be removed.
    
    """

    from MACS3.Signal.PeakDetect import PeakDetect
    from math import log
    import tempfile
    from exprmat.ansi import warning, info

    if isinstance(groupby, str):
        groupby = list(adata.obs[groupby])
    if replicate is not None and isinstance(replicate, str):
        replicate = list(adata.obs[replicate])

    # MACS3 options
    options = type('MACS3_OPT', (), {})()
    options.info = lambda _: None
    options.debug = lambda _: None
    options.warn = lambda x: warning(x)
    options.name = "MACS3"
    options.bdg_treat = 't'
    options.bdg_control = 'c'
    options.cutoff_analysis = False
    options.cutoff_analysis_file = 'a'
    options.store_bdg = False
    options.do_SPMR = False
    options.trackline = False
    options.log_pvalue = None
    options.log_qvalue = log(qvalue, 10) * -1
    options.PE_MODE = False

    options.gsize = adata.uns['assembly.size'].sum()
    options.maxgap = 30 # The maximum allowed gap between two nearby regions to be merged
    options.minlen = extsize if min_len is None else min_len
    options.shift = shift
    options.nolambda = nolambda
    options.smalllocal = 1000
    options.largelocal = 10000
    options.call_summits = False if call_broad_peaks else True
    options.broad = call_broad_peaks
    if options.broad: options.log_broadcutoff = log(broad_cutoff, 10) * -1

    options.fecutoff = 1.0
    options.d = extsize
    options.scanwindow = 2 * options.d

    if groupby is None:
        peaks = internal.call_peaks_bulk(adata, options, max_frag_size)
        if inplace:
            adata.uns[key_added + ".whole"] = peaks.to_pandas() if not adata.isbacked else peaks
            return
        else:
            return peaks

    with tempfile.TemporaryDirectory(dir=tempdir) as tmpdirname:
        
        info('exporting fragments ...')
        fragments = internal.export_tags(
            adata, tmpdirname, groupby, replicate, max_frag_size, selections)

        def _call_peaks(tags):

            import tempfile
            tempfile.tempdir = tmpdirname  # overwrite the default tempdir in MACS3
            merged, reps = internal.create_fwtrack_obj(tags)
            options.log_qvalue = log(qvalue, 10) * -1
            
            import logging
            logging.getLogger().setLevel(logging.CRITICAL + 1) # temporarily disable logging

            peakdetect = PeakDetect(treat = merged, opt = options)
            peakdetect.call_peaks()
            peakdetect.peaks.filter_fc(fc_low = options.fecutoff)
            merged = peakdetect.peaks

            others = []
            if replicate_qvalue is not None:
                options.log_qvalue = log(replicate_qvalue, 10) * -1
            for x in reps:
                peakdetect = PeakDetect(treat = x, opt = options)
                peakdetect.call_peaks()
                peakdetect.peaks.filter_fc(fc_low = options.fecutoff)
                others.append(peakdetect.peaks)
            
            logging.getLogger().setLevel(logging.INFO) # enable logging
            return internal.find_reproducible_peaks(merged, others, blacklist)

        info("calling peaks ...")
        if n_jobs == 1: peaks = [_call_peaks(x) for x in fragments.values()]
        else: peaks = parallel_map(_call_peaks, [(x,) for x in fragments.values()], n_jobs)
        peaks = {k: v for k, v in zip(fragments.keys(), peaks)}
        
        if inplace:
            if adata.isbacked: adata.uns[key_added] = peaks
            else: adata.uns[key_added] = {k: v.to_pandas() for k, v in peaks.items()}
        else: return peaks


def merge_peaks(
    peaks,
    chrom_sizes: dict[str, int],
    half_width: int = 250,
):
    """
    Merge peaks from different groups.

    Merge peaks from different groups. It is typically used to merge
    results from :func:`~snapatac2.tools.macs3`.

    This function initially expands the summits of identified peaks by `half_width`
    on both sides. Following this expansion, it addresses the issue of overlapping
    peaks through an iterative process. The procedure begins by prioritizing the
    most significant peak, determined by the smallest p-value. This peak is retained,
    and any peak that overlaps with it is excluded. Subsequently, the same method
    is applied to the next most significant peak. This iteration continues until
    all peaks have been evaluated, resulting in a final list of non-overlapping
    peaks, each with a fixed width determined by the initial extension.

    Parameters
    ----------
    peaks
        Peak information from different groups.

    half_width
        Half width of the merged peaks.
    
    Note
    ----
    For bulk ATAC-seq and ChIP-seq datasets, we recommend identify consensus peaks
    by IDR first, before merging the peaks to form matrices.
    """
    import pandas as pd
    import polars as pl

    peaks = { k: pl.from_pandas(v) if isinstance(v, pd.DataFrame) else v for k, v in peaks.items()}
    return internal.py_merge_peaks(peaks, chrom_sizes, half_width)


def parallel_map(mapper, args, nprocs):
    import time
    from multiprocess import get_context
    from tqdm import tqdm

    with get_context("spawn").Pool(nprocs) as pool:

        procs = set(pool._pool)
        jobs = [(i, pool.apply_async(mapper, x)) for i, x in enumerate(args)]
        results = []

        with tqdm(total = len(jobs)) as pbar:
            while len(jobs) > 0:
                if any(map(lambda p: not p.is_alive(), procs)):
                    raise RuntimeError("some worker process died unexpectedly.")

                remaining = []
                for i, job in jobs:
                    if job.ready():
                        results.append((i, job.get()))
                        pbar.update(1)
                    else: remaining.append((i, job))
                
                jobs = remaining
                time.sleep(0.5)

        return [x for _,x in sorted(results, key = lambda x: x[0])]