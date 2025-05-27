
import pandas as pd
from exprmat.ansi import warning, error, info
from exprmat.data.lr import (
    select_resource,
    handle_resource
)

from exprmat.lr.utils import (
    common_method_columns as M, 
    default_common_columns as C, 
    default_primary_columns as P, 
    internal_values as I
)


def filter_reassemble_complexes(
    lr_res, key_columns, complex_cols, expr_prop,
    return_all_lrs = False, complex_policy = 'min'
):
    
    # Filter by expr_prop (inner join only complexes where all subunits are expressed)
    expressed = (
        lr_res[key_columns + [C.ligand_props, C.receptor_props]]
            .set_index(key_columns)
            .stack()
            .groupby(key_columns)
            .agg(prop_min = complex_policy)
            .reset_index()
        )
    
    expressed = expressed.rename(columns = {'prop_min': I.prop_min})
    expressed = expressed[expressed[I.prop_min] >= expr_prop]

    if not return_all_lrs:
        lr_res = lr_res.merge(expressed, how = 'inner', on = key_columns)
    else:
        expressed[I.lrs_to_keep] = True
        lr_res = lr_res.merge(expressed, how = 'left', on = key_columns)
        # deal with duplicated subunits
        # subunits that are not expressed might not represent the most relevant subunit
        lr_res.drop_duplicates(subset = key_columns, inplace = True)
        lr_res[I.lrs_to_keep].fillna(value = False, inplace = True)
        lr_res[I.prop_min].fillna(value = 0, inplace = True)

    # check if complex policy is only min
    aggs = { complex_policy, 'min' }

    for col in complex_cols:
        lr_res = reduce_complexes(
            col = col, lr_res = lr_res, key_cols = key_columns, aggs = aggs
        )

    # check if there are any duplicated subunits
    duplicate_mask = lr_res.duplicated(subset = key_columns, keep = False)
    if duplicate_mask.any():
        # check if there are any non-equal subunit values
        if not lr_res[duplicate_mask].groupby(key_columns)[complex_cols].transform(
            lambda x: x.duplicated(keep = False)).all().all():
            warning('there were duplicated subunits in the complexes. ')
            warning('the subunits were reduced to only the minimum expression subunit. ')
            warning('however, there were subunits that were not the same within a complex. ')

        lr_res = lr_res.drop_duplicates(subset = key_columns, keep = 'first')

    return lr_res


def reduce_complexes(
    col: str, lr_res: pd.DataFrame,
    key_cols: list, aggs: (dict | str)
):
    lr_res = lr_res.groupby(key_cols)

    # get min cols by which we will join
    # then rename from agg name to column name (e.g. 'min' to 'ligand_min')
    lr_min = lr_res[col].agg(aggs).reset_index().copy(). \
        rename(columns = {agg: col.split('.')[0] + '.' + agg for agg in aggs})

    # right is the min subunit for that column
    join_key = col.split('.')[0] + '.min'  # ligand_min or receptor_min

    # Here, I join the min value and keep only those rows that match
    lr_res = lr_res.obj.merge(lr_min, on = key_cols, how = 'inner')
    lr_res = lr_res[lr_res[col] == lr_res[join_key]].drop(join_key, axis = 1)

    return lr_res


def explode_complexes(
    resource: pd.DataFrame,
    source = P.ligand, target = P.receptor
) -> pd.DataFrame:
    
    resource['interaction'] = resource[source] + '&' + resource[target]
    resource = (
        resource.set_index('interaction')
            .apply(lambda x: x.str.split('_'))
            .explode([target])
            .explode(source)
            .reset_index()
        )
    
    resource[[f'{source}.complex', f'{target}.complex']] = resource[
        'interaction'].str.split('&', expand=True)

    return resource