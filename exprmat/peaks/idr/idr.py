
import os, sys
import gzip, io
import math
import numpy
from scipy.stats.stats import rankdata

from collections import namedtuple, defaultdict, OrderedDict
from itertools import chain

def mean(items):
    items = list(items)
    return sum(items) / float(len(items))

import exprmat.peaks.idr.constants as idrc
from exprmat.peaks.idr.optimization import estimate_model_params
from exprmat.peaks.idr.utils import calc_post_membership_prbs, compute_pseudo_values
from exprmat.ansi import info, error

peak_columns = [
    'chr', 'strand', 'start', 'end', 'score', 
    'summit', 'signal', 'p', 'q'
]

Peak = namedtuple('Peak', peak_columns)

MergedPeak = namedtuple('Peak', [
    'chr', 'strand', 'start', 'end', 'summit', 
    'merged', 'signals', 'pks'
])


def correct_multi_summit_peak_idr(idr_values, merged_peaks):

    assert len(idr_values) == len(merged_peaks)
    new_values = idr_values.copy()
    # find the maximum IDR value for each peak
    pk_idr_values = defaultdict(lambda: float('inf')) 
    for i, pk in enumerate(merged_peaks):
        pk_idr_values[(pk.chr, pk.strand, pk.start, pk.end)] = min(
            pk_idr_values[(pk.chr, pk.strand, pk.start, pk.end)], 
            idr_values[i]
        )

    # store the indices best peak indices, and update the values
    best_indices = []
    for i, pk in enumerate(merged_peaks):
        region = (pk.chr, pk.strand, pk.start, pk.end)
        if new_values[i] == pk_idr_values[region]: best_indices.append(i)
        else: new_values[i] = pk_idr_values[region]

    return numpy.array(best_indices), new_values


def iter_merge_grpd_intervals(
    intervals, n_samples, pk_agg_fn,
    use_oracle_pks, use_nonoverlapping_peaks
):
    # grp peaks by their source, and calculate the merged peak boundaries
    grpd_peaks = OrderedDict([(i+1, []) for i in range(n_samples)])
    pk_start, pk_stop = 1e12, -1
    for interval, sample_id in intervals:
        # if we've provided a unified peak set, ignore any intervals that 
        # don't contain it for the purposes of generating the merged list
        if (not use_oracle_pks) or sample_id == 0:
            pk_start = min(interval.start, pk_start)
            pk_stop = max(interval.end, pk_stop)
        # if this is an actual sample (ie not a merged peaks)
        if sample_id > 0:
            grpd_peaks[sample_id].append(interval)
    
    # if there are no identified peaks, continue (this can happen if 
    # we have a merged peak list but no merged peaks overlap sample peaks)
    if pk_stop == -1:
        return None

    # skip regions that dont have a peak in all replicates
    if not use_nonoverlapping_peaks:
        if any(0 == len(peaks) for peaks in grpd_peaks.values()):
            return None

    # find the merged peak summit
    # note that we can iterate through the values because 
    # grpd_peaks is an ordered dict
    replicate_summits = []
    for sample_id, pks in grpd_peaks.items():
        # if an oracle peak set is specified, skip the replicates
        if use_oracle_pks and sample_id != 0: 
            continue

        # initialize the summit to the first peak
        try: replicate_summit, summit_signal = pks[0].summit, pks[0].score
        except IndexError: replicate_summit, summit_signal =  None, -1e9
        # if there are more peaks, take the summit that corresponds to the 
        # replicate peak with the highest signal value
        for pk in pks[1:]:
            if pk.summit != None and pk.score > summit_signal:
                replicate_summit, summit_signal = pk.summit, pk.score
        # make sure a peak summit was specified
        if replicate_summit != None:
            replicate_summits.append( replicate_summit )

    summit = ( int(mean(replicate_summits)) 
               if len(replicate_summits) > 0 else None )

    # note that we can iterate through the values because 
    # grpd_peaks is an ordered dict
    signals = [pk_agg_fn(pk.score for pk in pks) if len(pks) > 0 else 0
              for pks in grpd_peaks.values()]
    merged_pk = (pk_start, pk_stop, summit, 
                 pk_agg_fn(signals), signals, grpd_peaks)

    yield merged_pk
    return


def iter_matched_oracle_pks(
        pks, n_samples, pk_agg_fn, use_nonoverlapping_peaks=False ):
    """
    Match each oracle peak to it nearest replicate peaks.
    """

    oracle_pks = [pk for pk, sample_id in pks if sample_id == 0]
    # if there are zero oracle peaks in this 
    if len(oracle_pks) == 0: return None
    # for each oracle peak, find score the replicate peaks
    for oracle_pk in oracle_pks:
        peaks_and_scores = OrderedDict([(i+1, []) for i in range(n_samples)])
        for pk, sample_id in pks:
            # skip oracle peaks
            if sample_id == 0: continue
            
            # calculate the distance between summits, setting it to a large
            # value in case the peaks dont have summits
            summit_distance = sys.maxsize
            if oracle_pk.summit != None and pk.summit != None:
                summit_distance = abs(oracle_pk.summit - pk.summit)
            # calculate the fraction overlap witht he oracle peak
            overlap = (1 + min(oracle_pk.end, pk.end) 
                       - max(oracle_pk.start, pk.start) ) 
            overlap_frac = overlap/(oracle_pk.end - oracle_pk.start + 1)
            
            peaks_and_scores[sample_id].append(
                ((summit_distance, -overlap_frac, -pk.score), pk))
                
        # skip regions that dont have a peak in all replicates. 
        if not use_nonoverlapping_peaks and any(
                0 == len(peaks) for peaks in peaks_and_scores.values()):
            continue
        
        # build the aggregated signal value, which is jsut the signal value
        # of the replicate peak witgh the closest match
        signals = []
        rep_pks = []
        for rep_id, scored_pks in peaks_and_scores.items():
            scored_pks.sort()
            if len(scored_pks) == 0:
                assert use_nonoverlapping_peaks
                signals.append(0)
                rep_pks.append(None)
            else:
                signals.append(scored_pks[0][1].score)
                rep_pks.append( [scored_pks[0][1],] )

        all_peaks = [oracle_pk,] + rep_pks
        new_pk = (
            oracle_pk.start, oracle_pk.end, oracle_pk.summit, 
            pk_agg_fn(signals), 
            signals, 
            OrderedDict(zip(range(len(all_peaks)), all_peaks))
        )

        yield new_pk

    return


def merge_peaks_in_contig(
    all_s_peaks, pk_agg_fn, oracle_pks = None,
    use_nonoverlapping_peaks = False
):
    """
    Merge peaks in a single contig/strand.
    returns: The merged peaks. 
    """

    # merge and sort all peaks, keeping track of which sample they originated in
    oracle_pks_iter = []
    if oracle_pks != None: 
        oracle_pks_iter = oracle_pks
    
    # merge and sort all of the intervals, leeping track of their source
    all_intervals = []
    for sample_id, peaks in enumerate([oracle_pks_iter,] + all_s_peaks):
        all_intervals.extend((pk,sample_id) for pk in peaks)
    all_intervals.sort()
    
    # grp overlapping intervals. Since they're already sorted, all we need
    # to do is check if the current interval overlaps the previous interval
    grpd_intervals = [[],]
    curr_start, curr_stop = all_intervals[0][:2]
    for pk, sample_id in all_intervals:
        if pk.start < curr_stop:
            curr_stop = max(pk.end, curr_stop)
            grpd_intervals[-1].append((pk, sample_id))
        else:
            curr_start, curr_stop = pk.start, pk.end
            grpd_intervals.append([(pk, sample_id),])

    # build the unified peak list, setting the score to 
    # zero if it doesn't exist in both replicates
    merged_pks = []
    if oracle_pks == None:
        for intervals in grpd_intervals:
            for merged_pk in iter_merge_grpd_intervals(
                    intervals, len(all_s_peaks), pk_agg_fn,
                    use_oracle_pks=(oracle_pks != None),
                    use_nonoverlapping_peaks = use_nonoverlapping_peaks):
                merged_pks.append(merged_pk)
    else:        
        for intervals in grpd_intervals:
            for merged_pk in iter_matched_oracle_pks(
                    intervals, len(all_s_peaks), pk_agg_fn,
                    use_nonoverlapping_peaks = use_nonoverlapping_peaks):
                merged_pks.append(merged_pk)
    
    return merged_pks


def merge_peaks(
    all_s_peaks, pk_agg_fn, oracle_pks = None, 
    use_nonoverlapping_peaks = False
):
    """
    Merge peaks over all contig/strands
    """

    # if we have reference peaks, use its contigs: otherwise use
    # the union of the replicates contigs
    if oracle_pks != None: contigs = sorted(oracle_pks.keys())
    else: contigs = sorted(set(chain(*[list(s_peaks.keys()) for s_peaks in all_s_peaks])))

    merged_peaks = []
    for key in contigs:
        # check to see if we've been provided a peak list and, if so, 
        # pass it down. If not, set the oracle peaks to None so that 
        # the callee knows not to use them
        if oracle_pks != None: contig_oracle_pks = oracle_pks[key]
        else: contig_oracle_pks = None
        
        # since s*_peaks are default dicts, it will never raise a key error, 
        # but instead return an empty list which is what we want
        merged_contig_peaks = merge_peaks_in_contig(
            [s_peaks[key] for s_peaks in all_s_peaks], 
            pk_agg_fn, contig_oracle_pks, 
            use_nonoverlapping_peaks = use_nonoverlapping_peaks)
        merged_peaks.extend(
            MergedPeak(*(key + pk)) for pk in merged_contig_peaks)
    
    merged_peaks.sort(key=lambda x: x.merged, reverse = True)
    return merged_peaks


def build_rank_vectors(merged_peaks):

    # allocate memory for the ranks vector
    s1 = numpy.zeros(len(merged_peaks))
    s2 = numpy.zeros(len(merged_peaks))
    # add the signal
    for i, x in enumerate(merged_peaks):
        s1[i], s2[i] = x.signals

    rank1 = numpy.lexsort((numpy.random.random(len(s1)), s1)).argsort()
    rank2 = numpy.lexsort((numpy.random.random(len(s2)), s2)).argsort()
    
    return ( numpy.array(rank1, dtype = numpy.int32), 
             numpy.array(rank2, dtype = numpy.int32) )


def calc_local_idr(theta, r1, r2):
    
    mu, sigma, rho, p = theta
    z1 = compute_pseudo_values(r1, mu, sigma, p, EPS=1e-12)
    z2 = compute_pseudo_values(r2, mu, sigma, p, EPS=1e-12)
    localIDR = 1 - calc_post_membership_prbs(numpy.array(theta), z1, z2)
    if idrc.FILTER_PEAKS_BELOW_NOISE_MEAN: localIDR[z1 + z2 < 0] = 1 

    # it doesn't make sense for the IDR values to be smaller than the 
    # optimization tolerance
    localIDR = numpy.clip(localIDR, idrc.CONVERGENCE_EPS_DEFAULT, 1)
    return localIDR


def calc_global_idr(localIDR):

    local_idr_order = localIDR.argsort()
    ordered_local_idr = localIDR[local_idr_order]
    ordered_local_idr_ranks = rankdata(ordered_local_idr, method = 'max')
    IDR = []
    for i, rank in enumerate(ordered_local_idr_ranks):
        IDR.append(ordered_local_idr[:rank].mean())
    IDR = numpy.array(IDR)[local_idr_order.argsort()]
    return IDR


def fit_model_and_calc_local_idr(
    r1, r2, 
    starting_point = None,
    max_iter = idrc.MAX_ITER_DEFAULT, 
    convergence_eps = idrc.CONVERGENCE_EPS_DEFAULT, 
    fix_mu = False, fix_sigma = False
):
    
    # in theory we would try to find good starting point here,
    # but for now just set it to somethign reasonable
    if starting_point is None:
        starting_point = (
            idrc.DEFAULT_MU, idrc.DEFAULT_SIGMA,
            idrc.DEFAULT_RHO, idrc.DEFAULT_MIX_PARAM
        )
    

    # fit the model parameters    
    info("fitting the model parameters ...");
    
    theta, loss = estimate_model_params(
        r1, r2,
        starting_point, 
        max_iter = max_iter, 
        convergence_eps = convergence_eps,
        fix_mu = fix_mu, fix_sigma = fix_sigma
    )
    
    info("final parameter values: [%s]" % " ".join("%.2f" % x for x in theta))
    
    # calculate the global IDR
    local_idr = calc_local_idr(numpy.array(theta), r1, r2)
    return local_idr


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Program: IDR (Irreproducible Discovery Rate)
Version: {PACKAGE_VERSION}
Contact: Nathan Boley <npboley@gmail.com>
""".format(PACKAGE_VERSION=idrc.__version__))

    def PossiblyGzippedFile(fname):
        if fname.endswith(".gz"):
            return io.TextIOWrapper(gzip.open(fname, 'rb'))
        else:
            return open(fname, 'r')
    
    parser.add_argument( '--samples', '-s', type=PossiblyGzippedFile, nargs=2, 
                         required=True,
                         help='Files containing peaks and scores.')
    parser.add_argument( '--peak-list', '-p', type=PossiblyGzippedFile,
        help='If provided, all peaks will be taken from this file.')
    parser.add_argument( '--input-file-type', default='narrowPeak',
                         choices=['narrowPeak', 'broadPeak', 'bed', 'gff'], 
        help='File type of --samples and --peak-list.')
    
    parser.add_argument( '--rank',
        help="Which column to use to rank peaks."\
            +"\t\nOptions: signal.value p.value q.value columnIndex"\
            +"\nDefaults:\n\tnarrowPeak/broadPeak: signal.value\n\tbed: score")
    
    default_ofname = "idrValues.txt"
    parser.add_argument( '--output-file', "-o", 
                         default=default_ofname, 
        help='File to write output to.\nDefault: {}'.format(default_ofname))
    parser.add_argument( '--output-file-type', 
                         choices=['narrowPeak', 'broadPeak', 'bed'], 
                         default=None, 
        help='Output file type. Defaults to input file type when available, otherwise bed.')

    parser.add_argument( '--log-output-file', "-l", type=argparse.FileType("w"),
                         default=sys.stderr,
                         help='File to write output to. Default: stderr')
    
    parser.add_argument( '--idr-threshold', "-i", type=float, default=None,
        help="Only return peaks with a global idr threshold below this value."\
            +"\nDefault: report all peaks")
    parser.add_argument( '--soft-idr-threshold', type=float, default=None, 
        help="Report statistics for peaks with a global idr below this "\
        +"value but return all peaks with an idr below --idrc.\nDefault: %.2f" \
                         % idrc.DEFAULT_SOFT_IDR_THRESH)

    parser.add_argument( '--use-old-output-format', 
                         action='store_true', default=False,
                         help="Use old output format.")

    parser.add_argument( '--plot', action='store_true', default=False,
                         help='Plot the results to [OFNAME].png')
        
    parser.add_argument( '--use-nonoverlapping-peaks', 
                         action="store_true", default=False,
        help='Use peaks without an overlapping match and set the value to 0.')
    
    parser.add_argument( '--peak-merge-method', 
                         choices=["sum", "avg", "min", "max"], default=None,
        help="Which method to use for merging peaks.\n" \
              + "\tDefault: 'sum' for signal/score/column indexes, 'min' for p/q-value.")

    parser.add_argument( '--initial-mu', type=float, default=idrc.DEFAULT_MU,
        help="Initial value of mu. Default: %.2f" % idrc.DEFAULT_MU)
    parser.add_argument( '--initial-sigma', type=float, 
                         default=idrc.DEFAULT_SIGMA,
        help="Initial value of sigma. Default: %.2f" % idrc.DEFAULT_SIGMA)
    parser.add_argument( '--initial-rho', type=float, default=idrc.DEFAULT_RHO,
        help="Initial value of rho. Default: %.2f" % idrc.DEFAULT_RHO)
    parser.add_argument( '--initial-mix-param', 
        type=float, default=idrc.DEFAULT_MIX_PARAM,
        help="Initial value of the mixture params. Default: %.2f" \
                         % idrc.DEFAULT_MIX_PARAM)

    parser.add_argument( '--fix-mu', action='store_true', 
        help="Fix mu to the starting point and do not let it vary.")    
    parser.add_argument( '--fix-sigma', action='store_true', 
        help="Fix sigma to the starting point and do not let it vary.")    

    parser.add_argument( '--dont-filter-peaks-below-noise-mean', 
                         default=False,
                         action='store_true', 
        help="Allow signal points that are below the noise mean (should only be used if you know what you are doing).")    

    parser.add_argument( '--use-best-multisummit-IDR',
                         default=False, action='store_true',
                         help="Set the IDR value for a group of multi summit peaks (a group of peaks with the same chr/start/stop but different summits) to the best value across all of these peaks. This is a work around for peak callers that don't do a good job splitting scores across multi summit peaks (e.g. MACS). If set in conjunction with --plot two plots will be created - one with alternate summits and one without.  Use this option with care.")

    parser.add_argument( '--allow-negative-scores', 
                         default=False,
                         action='store_true', 
        help="Allow negative values for scores. (should only be used if you know what you are doing)")    

    parser.add_argument( '--random-seed', type=int, default=0, 
        help="The random seed value (sor braking ties). Default: 0") 
    parser.add_argument( '--max-iter', type=int, default=idrc.MAX_ITER_DEFAULT, 
        help="The maximum number of optimization iterations. Default: %i" 
                         % idrc.MAX_ITER_DEFAULT)
    parser.add_argument( '--convergence-eps', type=float, 
                         default=idrc.CONVERGENCE_EPS_DEFAULT, 
        help="The maximum change in parameter value changes " \
             + "for convergence. Default: %.2e" % idrc.CONVERGENCE_EPS_DEFAULT)
    
    parser.add_argument( '--only-merge-peaks', action='store_true', 
        help="Only return the merged peak list.")    
    
    parser.add_argument( '--verbose', action="store_true", default=False, 
                         help="Print out additional debug information")
    parser.add_argument( '--quiet', action="store_true", default=False, 
                         help="Don't print any status messages")

    parser.add_argument('--version', action='version', 
                        version='IDR %s' % idrc.__version__)

    args = parser.parse_args()

    args.output_file = open(args.output_file, "w")
    idrc.log_ofp = args.log_output_file

    if args.output_file_type is None:
        if args.input_file_type in ('narrowPeak', 'broadPeak', 'bed'):
            args.output_file_type = args.input_file_type
        else:
            args.output_file_type = 'bed'
    
    if args.verbose: 
        idrc.VERBOSE = True 

    global QUIET
    if args.quiet: 
        idrc.QUIET = True 
        idrc.VERBOSE = False

    if args.dont_filter_peaks_below_noise_mean is True:
        idrc.FILTER_PEAKS_BELOW_NOISE_MEAN = False

    if args.allow_negative_scores is True:
        idrc.ONLY_ALLOW_NON_NEGATIVE_VALUES = False
        
    assert idrc.DEFAULT_IDR_THRESH == 1.0
    if args.idr_threshold == None and args.soft_idr_threshold == None:
        args.idr_threshold = idrc.DEFAULT_IDR_THRESH
        args.soft_idr_threshold = idrc.DEFAULT_SOFT_IDR_THRESH
    elif args.soft_idr_threshold == None:
        assert args.idr_threshold != None
        args.soft_idr_threshold = args.idr_threshold
    elif args.idr_threshold == None:
        assert args.soft_idr_threshold != None
        args.idr_threshold = idrc.DEFAULT_IDR_THRESH

    numpy.random.seed(args.random_seed)

    if args.plot:
        try: 
            import matplotlib
        except ImportError:
            idrc.log("WARNING: matplotlib does not appear to be installed and "\
                    +"is required for plotting - turning plotting off.", 
                    level="WARNING" )
            args.plot = False
    
    return args

def load_samples(
    df1, df2,
    signal_type = 'score',
    summit_type = 'summit',
    peak_merge_fn = sum,
    use_nonoverlapping_peaks = False
):  
    def df_to_dict(df):
        f = defaultdict(list)
        for i_chrom in df['chr'].unique():
            for i_strand in df['strand'].unique():
                selection = df.loc[(df['chr'] == i_chrom) & (df['strand'] == i_strand), :]
                valist = []
                for xid in range(len(selection)):
                    prop = selection.iloc[xid, :]
                    valist.append(Peak(
                        prop['chr'], prop['strand'], prop['start'], prop['end'],
                        prop['score'], prop['summit'], prop['fc'], prop['p'], prop['q']
                    ))

                f[(i_chrom, i_strand)] = valist
        return f
    
    f1 = df_to_dict(df1)
    f2 = df_to_dict(df2)
      
    info('merging peaks ...')
    merged_peaks = merge_peaks(
        [f1, f2], peak_merge_fn, 
        use_nonoverlapping_peaks = use_nonoverlapping_peaks
    )
    return merged_peaks, signal_type


def idr(
    df1, df2,
    signal_type: str = 'score',
    summit_type: str = 'summit',
    peak_merge_fn = sum,
    use_nonoverlapping_peaks: bool = False,
    filter_peaks_below_noise = True,
    allow_negative_scores = False,
    idr_threshold = None,
    soft_idr_threshold = None,
    random_seed = 42,
    only_merge_peaks = False,
    use_best_multisummit_idr = False,
    initial_mu = idrc.DEFAULT_MU,
    initial_rho = idrc.DEFAULT_RHO,
    initial_sigma = idrc.DEFAULT_SIGMA,
    initial_mix_param = idrc.DEFAULT_MIX_PARAM,
    max_iter = idrc.MAX_ITER_DEFAULT,
    convergence_eps = idrc.CONVERGENCE_EPS_DEFAULT,
    fix_mu = False,
    fix_sigma = False,
    filter_global_idr = None
):

    idrc.FILTER_PEAKS_BELOW_NOISE_MEAN = filter_peaks_below_noise
    idrc.ONLY_ALLOW_NON_NEGATIVE_VALUES = not allow_negative_scores

    assert idrc.DEFAULT_IDR_THRESH == 1.0
    if idr_threshold == None and soft_idr_threshold == None:
        idr_threshold = idrc.DEFAULT_IDR_THRESH
        soft_idr_threshold = idrc.DEFAULT_SOFT_IDR_THRESH
    elif soft_idr_threshold == None:
        assert idr_threshold != None
        soft_idr_threshold = idr_threshold
    elif idr_threshold == None:
        assert soft_idr_threshold != None
        idr_threshold = idrc.DEFAULT_IDR_THRESH

    numpy.random.seed(random_seed)

    # load and merge peaks
    merged_peaks, signal_type = load_samples(
        df1, df2,
        signal_type = signal_type,
        summit_type = summit_type,
        peak_merge_fn = peak_merge_fn,
        use_nonoverlapping_peaks = use_nonoverlapping_peaks,
    )

    s1 = numpy.array([pk.signals[0] for pk in merged_peaks])
    s2 = numpy.array([pk.signals[1] for pk in merged_peaks])

    # build the ranks vector
    info('ranking peaks ...')
    r1, r2 = build_rank_vectors(merged_peaks)
    
    if only_merge_peaks:
        locidr, idrs = None, None
    
    else:

        if len(merged_peaks) < 20:
            error('should be at least 20 peaks after merging.')

        locidr = fit_model_and_calc_local_idr(
            r1, r2, 
            starting_point = (
                initial_mu, initial_sigma, 
                initial_rho, initial_mix_param
            ),
            max_iter = max_iter,
            convergence_eps = convergence_eps,
            fix_mu = fix_mu, fix_sigma = fix_sigma 
        )    

        # if the use chose to use the best multi summit IDR, then
        # make the correction and plot just the corrected peaks
        if use_best_multisummit_idr:
            update_indices, locidr = correct_multi_summit_peak_idr(
                locidr, merged_peaks
            )

            idrs = calc_global_idr(locidr)        
        # we wrap this in an else statement to avoid calculating the global IDRs twice
        else: idrs = calc_global_idr(locidr)

    # build dataframe from idr result
    dfdict = {
        'chr': [],
        'start': [],
        'end': [],
        'strand': [],
        'score': [],
        'fc': [],
        'p': [],
        'q': [],
        'summit': [],
        'signal.x': [],
        'signal.y': []
    }

    from numpy import mean
    for v in merged_peaks:
        dfdict['chr'].append(v.chr)
        dfdict['start'].append(v.start)
        dfdict['end'].append(v.end)
        dfdict['strand'].append(v.strand)

        if (len(v.pks[1]) > 0) and (len(v.pks[2]) > 0):
            dfdict['score'].append(peak_merge_fn([v.pks[1][0].score, v.pks[2][0].score]))
            dfdict['fc'].append(peak_merge_fn([v.pks[1][0].signal, v.pks[2][0].signal]))
            dfdict['p'].append(max(v.pks[1][0].p, v.pks[2][0].p))
            dfdict['q'].append(max(v.pks[1][0].q, v.pks[2][0].q))
            dfdict['signal.x'].append(v.pks[1][0][peak_columns.index(signal_type)])
            dfdict['signal.y'].append(v.pks[2][0][peak_columns.index(signal_type)])
        
        elif len(v.pks[1]) > 0:
            dfdict['score'].append(v.pks[1][0].score)
            dfdict['fc'].append(v.pks[1][0].signal)
            dfdict['p'].append(v.pks[1][0].p)
            dfdict['q'].append(v.pks[1][0].q)
            dfdict['signal.x'].append(float('nan'))
            dfdict['signal.y'].append(float('nan'))
        
        elif len(v.pks[2]) > 0:
            dfdict['score'].append(v.pks[2][0].score)
            dfdict['fc'].append(v.pks[2][0].signal)
            dfdict['p'].append(v.pks[2][0].p)
            dfdict['q'].append(v.pks[2][0].q)
            dfdict['signal.x'].append(float('nan'))
            dfdict['signal.y'].append(float('nan'))

        dfdict['summit'].append(v.summit)

    import pandas as pd
    df = pd.DataFrame(dfdict)
    if (locidr is not None) and (idrs is not None):
        df['idr.local'] = locidr
        df['idr.global'] = idrs
        if filter_global_idr is not None:
            return df.loc[df['idr.global'] <= filter_global_idr, :].copy()

    return df
