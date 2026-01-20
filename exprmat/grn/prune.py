
import os
import pickle
import tempfile
from functools import partial, reduce
from itertools import chain, repeat
from typing import Optional, Sequence, Type
import numpy as np
from math import ceil

# using dill package for pickling to avoid strange bugs.
from multiprocessing import cpu_count
from operator import concat
from typing import Callable, Sequence, Type, TypeVar

import pandas as pd
from boltons.iterutils import chunked_iter
from dask.dataframe.utils import make_meta
from multiprocessing_on_dill.connection import Pipe
from multiprocessing_on_dill.context import Process

from exprmat.data.signature import signature, regulon
from exprmat.data.cistarget import inmemory, ranking_db
from exprmat.descriptive.recovery import aucs as calc_aucs
from exprmat.descriptive.recovery import leading_edge_row, recovery
from exprmat.ansi import error, warning, info
from exprmat.grn.modules import (
    ACTIVATING_MODULE,
    COLUMN_NAME_ANNOTATION,
    COLUMN_NAME_MOTIF_ID,
    COLUMN_NAME_MOTIF_SIMILARITY_QVALUE,
    COLUMN_NAME_ORTHOLOGOUS_IDENTITY,
    COLUMN_NAME_TF,
    REPRESSING_MODULE,
)

COLUMN_NAME_NES = "nes"
COLUMN_NAME_AUC = "auc"
COLUMN_NAME_CONTEXT = "context"
COLUMN_NAME_TARGET_GENES = "targets"
COLUMN_NAME_RANK_AT_MAX = "rank.max"
COLUMN_NAME_TYPE = "type"

DF_META_DATA = make_meta(
    {
        ("enrichment", COLUMN_NAME_AUC): np.float64,
        ("enrichment", COLUMN_NAME_NES): np.float64,
        ("enrichment", COLUMN_NAME_MOTIF_SIMILARITY_QVALUE): np.float64,
        ("enrichment", COLUMN_NAME_ORTHOLOGOUS_IDENTITY): np.float64,
        ("enrichment", COLUMN_NAME_ANNOTATION): object,
        ("enrichment", COLUMN_NAME_CONTEXT): object,
        ("enrichment", COLUMN_NAME_TARGET_GENES): object,
        ("enrichment", COLUMN_NAME_RANK_AT_MAX): np.int64,
    },
    
    index = pd.MultiIndex.from_arrays(
        [[], []], names = (COLUMN_NAME_TF, COLUMN_NAME_MOTIF_ID)
    ),
)


def auc1st(
    db: Type[ranking_db],
    module: regulon,
    motif_annotations: pd.DataFrame,
    rank_threshold: int = 1500,
    auc_threshold: float = 0.05,
    nes_threshold = 3.0,
    weighted_recovery = False,
    filter_for_annotation = True,
):
    
    # load rank of genes from database.
    df = db.load(module)
    features, genes, rankings = df.index.values, df.columns.values, df.values
    weights = (
        np.asarray([module[gene] for gene in genes])
        if weighted_recovery
        else np.ones(len(genes))
    )

    # include check for modules with no genes that could be mapped to the db. 
    # this can happen when including non protein-coding genes in the expression matrix.
    if df.empty:
        info('no genes can be mapped to module. skipped.')
        return pd.DataFrame(), None, None, genes, None

    # calculate recovery curves, auc and nes values.
    # for fast unweighted implementation so weights to none.
    aucs = calc_aucs(df, db.n_genes, weights, auc_threshold)
    ness = (aucs - aucs.mean()) / aucs.std()

    # keep only features that are enriched, i.e. nes sufficiently high.
    enriched_features_idx = ness >= nes_threshold
    enriched_features = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            list(zip(repeat(module.transcription_factor), features[enriched_features_idx])),
            names=[COLUMN_NAME_TF, COLUMN_NAME_MOTIF_ID],
        ),
        data = {
            COLUMN_NAME_NES: ness[enriched_features_idx],
            COLUMN_NAME_AUC: aucs[enriched_features_idx],
        },
    )

    if len(enriched_features) == 0:
        return pd.DataFrame(), None, None, genes, None

    # find motif annotations for enriched features.
    annotated_features = pd.merge(
        enriched_features,
        motif_annotations,
        how = "left",
        left_index = True,
        right_index = True,
    )

    # when using cluster db annotations, keep direct if available 
    # otherwise use other (extended)
    annotated_features = annotated_features.sort_values(
        [COLUMN_NAME_MOTIF_SIMILARITY_QVALUE, COLUMN_NAME_ORTHOLOGOUS_IDENTITY],
        ascending=[False, True],
    )

    annotated_features = annotated_features[
        ~ annotated_features.index.duplicated(keep = "last")
    ]

    annotated_features_idx = (
        pd.notnull(annotated_features[COLUMN_NAME_ANNOTATION])
        if filter_for_annotation
        else np.full((len(enriched_features),), True)
    )

    if len(annotated_features[annotated_features_idx]) == 0:
        return pd.DataFrame(), None, None, genes, None

    # calculated leading edge for the remaining enriched features that have annotations. 
    # the leading edge is calculated based on the average recovery curve so we still no 
    # to calculate all recovery curves. currently this is done via preallocating memory. 
    # this introduces a huge burden on memory when using region-based databases and multiple 
    # cores on a cluster node. e.g.
    #
    #   (24,000 features * 25,000 rank_threshold * 8 bytes)/(1,024*1,024*1,024) = 4,4gb
    #   this creates a potential peak on memory of 48 cores * 4,4gb = 214 gb
    #
    # TODO: solution could be to go for an iterative approach boosted by numba. but before 
    # doing so investigate the broader issue with creep in memory usage when using the dask 
    # framework: use a memory profile tool (https://pythonhosted.org/pympler/muppy.html) to 
    # check what is kept in memory in all subprocesses/workers.

    rccs, _ = recovery(
        df, db.n_genes, weights, rank_threshold, auc_threshold, no_auc = True
    )

    avgrcc = rccs.mean(axis = 0)
    avg2stdrcc = avgrcc + 2.0 * rccs.std(axis = 0)

    rccs = rccs[enriched_features_idx, :][annotated_features_idx, :]
    rankings = rankings[enriched_features_idx, :][annotated_features_idx, :]

    # Add additional information to the dataframe.
    annotated_features = annotated_features[annotated_features_idx]
    context = frozenset(chain(module.context, [db.name]))
    annotated_features[COLUMN_NAME_CONTEXT] = len(annotated_features) * [context]

    return annotated_features, rccs, rankings, genes, avg2stdrcc


default_module_to_features = partial(
    auc1st,
    rank_threshold = 1500,
    auc_threshold = 0.05,
    nes_threshold = 3.0,
    filter_for_annotation = True,
)


def module_to_df(
    db: Type[ranking_db],
    module: regulon,
    motif_annotations: pd.DataFrame,
    weighted_recovery = False,
    return_recovery_curves = False,
    module2features_func = default_module_to_features,
    verbose = False
) -> pd.DataFrame:
    
    try:
        df_annotated_features, rccs, rankings, genes, avg2stdrcc = module2features_func(
            db, module, motif_annotations, weighted_recovery = weighted_recovery
        )
    except MemoryError: error('(auc1st) out of memory.')

    # if less than 80% of the genes are mapped to the ranking database, 
    # the module is skipped.

    n_missing = len(module) - len(genes)
    frac_missing = float(n_missing) / len(module)
    if frac_missing >= 0.20:
        if verbose:
            warning(
                "less than 80 pct. of the genes in {} could be mapped to {}. skipped."
                .format(module.name, db.name)
            )

        return DF_META_DATA

    # if no annotated enriched features could be found, skip module.
    if len(df_annotated_features) == 0:
        return DF_META_DATA
    rank_threshold = rccs.shape[1]

    # combine elements into a dataframe.
    df_annotated_features.columns = pd.MultiIndex.from_tuples(
        list(zip(repeat("enrichment"), df_annotated_features.columns))
    )

    df_rnks = pd.DataFrame(
        index = df_annotated_features.index,
        columns = pd.MultiIndex.from_tuples(list(zip(repeat("ranking"), genes))),
        data = rankings,
    )

    df_rccs = pd.DataFrame(
        index = df_annotated_features.index,
        columns = pd.MultiIndex.from_tuples(
            list(zip(repeat("recovery"), np.arange(rank_threshold)))),
        data = rccs,
    )

    df = pd.concat([df_annotated_features, df_rccs, df_rnks], axis=1)

    # calculate the leading edges for each row. always return importance 
    # from gene inference phase.

    weights = np.array([module[gene] for gene in genes])
    df[[
        ("enrichment", COLUMN_NAME_TARGET_GENES),
        ("enrichment", COLUMN_NAME_RANK_AT_MAX),
    ]] = df.apply(
        partial(leading_edge_row, avg2stdrcc = avg2stdrcc, genes = genes, weights = weights),
        axis=1,
    )

    del df["ranking"]
    if not return_recovery_curves:
        del df["recovery"]
        assert all(
            [col in df.columns for col in DF_META_DATA]
        ), f"column comparison to expected metadata failed. found:\n{df.columns}"
        return df[DF_META_DATA.columns]
    else: return df
    

def modules_to_df(
    db: Type[ranking_db],
    modules: Sequence[regulon],
    motif_annotations: pd.DataFrame,
    weighted_recovery=False,
    return_recovery_curves = False,
    module2features_func = default_module_to_features,
    verbose = False
) -> pd.DataFrame:
    
    # make sure return recovery curves is always set to false because the metadata 
    # for the distributed dataframe needs to be fixed for the dask framework.
    return pd.concat([
        module_to_df(
            db, module, motif_annotations,
            weighted_recovery, False, module2features_func,
            verbose = verbose
        ) for module in modules
    ])


def distributed(
    rnkdbs: Sequence[Type[ranking_db]],
    modules: Sequence[Type[signature]],
    motif_annotations_fname: str,
    transform_func,
    aggregate_func,
    motif_similarity_fdr: float = 0.001,
    orthologuous_identity_threshold: float = 0.0,
    num_workers = None,
    module_chunksize = 100,
    verbose = False
):

    # this implementation overcomes the i/o-bounded performance. each worker (subprocess) 
    # loads a dedicated ranking database and motif annotation table into its own memory 
    # space before consuming module. the implementation of each worker uses the auc-first 
    # numba jit based implementation of the algorithm.

    assert (
        len(rnkdbs) <= num_workers if num_workers else cpu_count()
    ), "the number of databases is larger than the number of cores."

    amplifier = int((num_workers if num_workers else cpu_count()) / len(rnkdbs))
    info("using {} workers.".format(len(rnkdbs) * amplifier))
    receivers = []

    for db in rnkdbs:
        for idx, chunk in enumerate(
            chunked_iter(modules, ceil(len(modules) / float(amplifier)))
        ):
            sender, receiver = Pipe()
            receivers.append(receiver)
            Worker(
                "{}({})".format(db.name, idx + 1),
                db,
                chunk,
                motif_annotations_fname,
                sender,
                motif_similarity_fdr,
                orthologuous_identity_threshold,
                transform_func,
                verbose = verbose
            ).start()

    # retrieve the name of the temporary file to which the data is stored.
    # this is a blocking operation.
    fnames = [recv.recv() for recv in receivers]
    # load all data from disk and concatenate.
    def load(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    try:
        return aggregate_func(list(map(load, fnames)))
    finally:
        # remove temporary files.
        for fname in fnames:
            os.remove(fname)


def load_motif_annotations(
    fname: str,
    column_names = (
        "#motif_id",
        "gene_name",
        "motif_similarity_qvalue",
        "orthologous_identity",
        "description",
    ),
    motif_similarity_fdr: float = 0.001,
    orthologous_identity_threshold: float = 0.0,
) -> pd.DataFrame:
    
    df = pd.read_csv(fname, sep = "\t", index_col=[1, 0], usecols = column_names)
    df.index.names = [COLUMN_NAME_TF, COLUMN_NAME_MOTIF_ID]
    df.rename(
        columns = {
            "motif_similarity_qvalue": COLUMN_NAME_MOTIF_SIMILARITY_QVALUE,
            "orthologous_identity": COLUMN_NAME_ORTHOLOGOUS_IDENTITY,
            "description": COLUMN_NAME_ANNOTATION,
        },
        inplace = True,
    )

    df = df[
        (df[COLUMN_NAME_MOTIF_SIMILARITY_QVALUE] <= motif_similarity_fdr) & 
        (df[COLUMN_NAME_ORTHOLOGOUS_IDENTITY] >= orthologous_identity_threshold)
    ]

    return df


class Worker(Process):
    def __init__(
        self,
        name: str,
        db: Type[ranking_db],
        modules: Sequence[regulon],
        motif_annotations_fname: str,
        sender,
        motif_similarity_fdr: float,
        orthologuous_identity_threshold: float,
        transformation_func,
        verbose = False,
    ):
        super().__init__(name = name)
        self.database = db
        self.modules = modules
        self.motif_annotations_fname = motif_annotations_fname
        self.motif_similarity_fdr = motif_similarity_fdr
        self.orthologuous_identity_threshold = orthologuous_identity_threshold
        self.transform_fnc = transformation_func
        self.sender = sender
        self.verbose = verbose

    def run(self):

        # info(f'job [{self.name}] started.')
        # Load ranking database in memory.
        rnkdb = inmemory(self.database)
        # Load motif annotations in memory.
        motif_annotations = load_motif_annotations(
            self.motif_annotations_fname,
            motif_similarity_fdr = self.motif_similarity_fdr,
            orthologous_identity_threshold = self.orthologuous_identity_threshold,
        )

        output = self.transform_fnc(
            rnkdb, self.modules, motif_annotations = motif_annotations,
            verbose = self.verbose
        )

        # info(f'job [{self.name}] finished.')

        # sending information back to parent process: to avoid overhead of pickling 
        # the data, the output is first written to disk in binary pickle format to a 
        # temporary file. The name of that file is shared with the parent process.

        output_fname = tempfile.mktemp()
        with open(output_fname, "wb") as f:
            pickle.dump(output, f)
        del output
        self.sender.send(output_fname)
        self.sender.close()


def prune(
    rnkdbs: Sequence[Type[ranking_db]],
    modules: Sequence[regulon],
    motif_annotations_fname: str,
    rank_threshold: int = 1500,
    auc_threshold: float = 0.05,
    nes_threshold = 3.0,
    motif_similarity_fdr: float = 0.001,
    orthologuous_identity_threshold: float = 0.0,
    weighted_recovery = False,
    num_workers = None,
    module_chunksize = 100,
    filter_for_annotation = True,
    verbose = False
) -> pd.DataFrame:
    
    """
    Calculate all regulons for a given sequence of ranking databases and a sequence of 
    co-expression modules. The number of regulons derived from the supplied modules is 
    usually much lower. In addition, the targets of the retained modules is reduced to 
    only these ones for which a cis-regulatory footprint is present.

    :param rnkdbs: The sequence of databases.
    :param modules: The sequence of modules.
    :param motif_annotations_fname: The name of the file that contains the motif annotations to use.
    :param rank_threshold: The total number of ranked genes to take into account when 
        creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for 
        the calculation of the area under the recovery curve.
    :param nes_threshold: The NES threshold to select enriched features.
    :param motif_similarity_fdr: The maximum False Discovery Rate to find factor 
        annotations for enriched motifs.
    :param orthologuous_identity_threshold: The minimum orthologuous identity to find 
        factor annotations for enriched motifs.
    :param weighted_recovery: Use weights of a gene signature when calculating recovery.
    """

    # always use module2features_auc1st_impl not only because of speed impact 
    # but also because of reduced memory footprint.
    module2features_func = partial(
        auc1st,
        rank_threshold = rank_threshold,
        auc_threshold = auc_threshold,
        nes_threshold = nes_threshold,
        filter_for_annotation = filter_for_annotation,
    )

    transformation_func = partial(
        modules_to_df,
        module2features_func = module2features_func,
        weighted_recovery = weighted_recovery,
    )

    # create a distributed dataframe from individual delayed objects 
    # to avoid out of memory problems.
    aggregation_func = pd.concat

    return distributed(
        rnkdbs,
        modules,
        motif_annotations_fname,
        transformation_func,
        aggregation_func,
        motif_similarity_fdr,
        orthologuous_identity_threshold,
        num_workers,
        module_chunksize,
        verbose = verbose
    )
