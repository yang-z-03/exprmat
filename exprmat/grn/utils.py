
from collections.abc import Sequence
from functools import reduce
import math

from exprmat.data.signature import regulon
from exprmat.ansi import error, info
from exprmat.grn.modules import (
    COLUMN_NAME_TF,
    COLUMN_NAME_MOTIF_ID,
    COLUMN_NAME_MOTIF_SIMILARITY_QVALUE,
    COLUMN_NAME_ORTHOLOGOUS_IDENTITY,
    COLUMN_NAME_ANNOTATION,

    ACTIVATING_MODULE,
    REPRESSING_MODULE,

    COLUMN_NAME_TARGET,
    COLUMN_NAME_WEIGHT,
    COLUMN_NAME_REGULATION,
    COLUMN_NAME_CORRELATION,
    RHO_THRESHOLD,
)

from exprmat.grn.prune import (
    COLUMN_NAME_NES,
    COLUMN_NAME_AUC,
    COLUMN_NAME_CONTEXT,
    COLUMN_NAME_TARGET_GENES,
    COLUMN_NAME_RANK_AT_MAX,
    COLUMN_NAME_TYPE,
)


def df_to_regulons(df, save_columns = []) -> Sequence[regulon]:
    
    if df.empty: error("signatures dataframe is empty.")
    info("creating regulons from a dataframe of enriched features.")
    info("additional columns saved: {}".format(save_columns))

    # because the code below will alter the dataframe we need to make a 
    # defensive copy of it.
    df = df.copy()

    # normally the columns index has two levels. For convenience of the 
    # following code the first level is removed.
    if df.columns.nlevels == 2:
        df.columns = df.columns.droplevel(0)

    def get_type(row):
        ctx = row[COLUMN_NAME_CONTEXT]
        # activating is the default!
        return REPRESSING_MODULE if REPRESSING_MODULE in ctx else ACTIVATING_MODULE

    df[COLUMN_NAME_TYPE] = df.apply(get_type, axis = 1)

    # group all rows per TF and type (+)/(-). 
    # each group results in a single regulon.
    not_none = lambda r: r is not None
    return list(
        filter(not_none, (
            regulon_for_group(tf_name, frozenset([interaction_type]), df_grp, save_columns)
            for (tf_name, interaction_type), df_grp in df.groupby(
                by=[COLUMN_NAME_TF, COLUMN_NAME_TYPE]
            ))
        )
    )


def regulon_for_group(tf_name, context, df_group, save_columns=[]) -> regulon:

    def score(nes, motif_similarity_qval, orthologuous_identity):
        
        # the combined score starts from the nes score which is then corrected for 
        # less confidence in the tf annotation in two steps:
        # 1. the orthologous identifity (a fraction between 0 and 1.0) is used directly to normalize the nes.
        # 2. the motif similarity q-value is converted to a similar fraction: -log10(q-value)
        # a motif that is directly annotated for the tf in the correct species is not penalized.

        correction_fraction = 1.0
        try:
            max_value = 10  # A q-value smaller than 10**-10 is considered the same as a q-value of 0.0.
            correction_fraction = (
                min(-math.log(motif_similarity_qval, 10), max_value) / max_value
                if not math.isnan(motif_similarity_qval) else 1.0
            )

        except ValueError: pass
        score = nes * correction_fraction

        # we assume that a non existing orthologous identity signifies a direct annotation.
        return (
            score if math.isnan(orthologuous_identity)
            else score * orthologuous_identity
        )

    def derive_interaction_type(ctx):
        return "(-)" if REPRESSING_MODULE in ctx else "(+)"

    def row_to_regulon(row):
        
        return regulon(
            name = "{} {}".format(tf_name, derive_interaction_type(context)),
            score = score(
                row[COLUMN_NAME_NES],
                row[COLUMN_NAME_MOTIF_SIMILARITY_QVALUE],
                row[COLUMN_NAME_ORTHOLOGOUS_IDENTITY],
            ),
            context = context,
            transcription_factor = tf_name,
            gene2weight = row[COLUMN_NAME_TARGET_GENES],
            gene2occurrence = [],
        )

    # find most enriched annotated motif and add this to the context
    df_selected = df_group.sort_values(by = COLUMN_NAME_NES, ascending = False)
    first_result_by_nes = df_selected.head(1).reset_index()

    # additional columns to the regulon
    nes = (
        first_result_by_nes[COLUMN_NAME_NES].values[0]
        if COLUMN_NAME_NES in save_columns else 0.0
    )

    orthologous_identity = (
        first_result_by_nes[COLUMN_NAME_ORTHOLOGOUS_IDENTITY].values[0]
        if COLUMN_NAME_ORTHOLOGOUS_IDENTITY in save_columns else 0.0
    )

    similarity_qvalue = (
        first_result_by_nes[COLUMN_NAME_MOTIF_SIMILARITY_QVALUE].values[0]
        if COLUMN_NAME_MOTIF_SIMILARITY_QVALUE in save_columns else 0.0
    )

    annotation = (
        first_result_by_nes[COLUMN_NAME_ANNOTATION].values[0]
        if COLUMN_NAME_ANNOTATION in save_columns else ""
    )

    # first we create a regulon for each enriched and annotated feature and then 
    # we aggregate these regulons into a single one using the union operator. this 
    # operator combined all target genes into a single set of genes keeping the 
    # maximum weight associated with a gene. in addition, the maximum combined 
    # score is kept as the score of the entire regulon.

    return reduce(
        regulon.union, (row_to_regulon(row) for _, row in df_group.iterrows())
    ).copy(
        context = frozenset(set(context)),
        nes = nes,
        orthologous_identity = orthologous_identity,
        similarity_qvalue = similarity_qvalue,
        annotation = annotation,
    )