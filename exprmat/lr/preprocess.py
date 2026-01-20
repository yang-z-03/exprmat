
import numpy as np
from anndata import AnnData
from typing import Optional
from pandas import DataFrame, Index
import scanpy as sc
from scipy.sparse import csr_matrix, isspmatrix_csr

from exprmat.utils import choose_layer
from exprmat.ansi import error, warning
from exprmat.lr.utils import default_params as P, internal_values as I


def assert_covered(
    subset, superset,
    subset_name: str = "resource",
    superset_name: str = "var_names",
    prop_missing_allowed: float = 0.98,
    verbose: bool = False
) -> None:

    subset = np.asarray(subset)
    is_missing = ~ np.isin(subset, superset)
    if subset.size == 0:
        prop_missing = 1.
        x_missing = 'values in interactions argument'
    
    else:
        prop_missing = np.sum(is_missing) / len(subset)
        x_missing = ", ".join([x for x in subset[is_missing]])
    
    if prop_missing > prop_missing_allowed:
        warning(f"please check if appropriate organism/id type was provided.")
        warning(
            f"allowed proportion ({prop_missing_allowed}) of missing "
            f"{subset_name} elements exceeded ({prop_missing:.2f}). "
        )
        error(f"too few features from the resource were found in the data.")


def prep_check_adata(
    adata: AnnData,
    groupby: (str | None),
    min_cells: (int | None),
    groupby_subset = None,
    use_raw: Optional[bool] = False,
    layer: Optional[str] = None,
    obsm = None,
    uns = None,
    complex_sep = P.complex_sep
) -> AnnData:
    
    X = choose_layer( adata = adata, use_raw = use_raw, layer = layer)
    if use_raw & (layer is None):
        var = DataFrame(index = adata.raw.var_names)
    else: var = DataFrame(index = adata.var_names)

    if obsm is not None:
        # discard any instances of AnnData if in obsm
        obsm = { k: v for k, v in obsm.items() if not isinstance(v, AnnData) }

    adata = sc.AnnData(
        X = X.astype('float32'), obs = adata.obs.copy(),
        var = var, obsp = adata.obsp.copy(), uns = uns, obsm = obsm
    ).copy()
    adata.var_names_make_unique()

    # check for empty features
    msk_features = np.sum(adata.X, axis = 0).A1 == 0
    n_empty_features = np.sum(msk_features)
    if n_empty_features > 0:
        warning(f"{n_empty_features} features are empty, they will be removed.")
        adata = adata[:, ~ msk_features]

    # check for empty samples
    msk_samples = adata.X.sum(axis = 1).A1 == 0
    n_empty_samples = np.sum(msk_samples)
    if n_empty_samples > 0:
        warning(f"{n_empty_samples} samples of mat are empty, they will be removed.")

    # check if log-norm
    _sum = np.sum(adata.X.data[0:100])
    if _sum == np.floor(_sum): # log normed values can seldem be integer.
        warning("you seem to pass integral values as input. we require log normalized data.")
        warning("check your data again, and ignore this if you confirmed.")

    # check for non-finite values
    if np.any(~ np.isfinite(adata.X.data)):
        error("expression matrix contains non finite values (nan or inf), please set them to 0 or remove them.")

    if groupby is not None:
        check_groupby(adata, groupby)

        if groupby_subset is not None:
            adata = adata[adata.obs[groupby].isin(groupby_subset), :]

        adata.obs[I.label] = adata.obs[groupby]

        # remove any cell types below X number of cells per cell type
        count_cells = adata.obs.groupby(groupby, observed = False)[groupby] \
            .size().reset_index(name = 'count').copy()
        count_cells['keep'] = count_cells['count'] >= min_cells

        if not all(count_cells.keep):
            lowly_abundant_idents = list(count_cells[~count_cells.keep][groupby])
            # remove lowly abundant identities
            msk = ~np.isin(adata.obs[[groupby]], lowly_abundant_idents)
            adata = adata[msk]
            warning(
                "the following cell identities were excluded: [{0}]"
                .format(", ".join(lowly_abundant_idents)),
            )

    # if genes already contains separaters to identify protein comples (here, '_')
    # it will be mis-regarded. gene names should only contain a-zA-Z0-9\-
    check_vars(
        adata.var_names,
        complex_sep = complex_sep,
    )

    # re-order adata vars alphabetically
    adata = adata[:, np.sort(adata.var_names)]
    return adata


def check_vars(var_names, complex_sep) -> list:

    var_issues = []
    if complex_sep is not None:
        for name in var_names:
            if complex_sep in name:
                var_issues.append(name)
    else: pass

    if len(var_issues) > 0:
        warning(f"{var_issues} contain `{complex_sep}`. you should replace those.")


def check_groupby(adata, groupby):
    if groupby not in adata.obs.columns:
        error(f"`{groupby}` not found in obs.columns.")
    if not adata.obs[groupby].dtype.name == 'category':
        error(f"converting `{groupby}` to categorical.")
        adata.obs[groupby] = adata.obs[groupby].astype('category')


def filter_resource(resource: DataFrame, var_names: Index) -> DataFrame:
    """
    Filter interactions for which vars are not present.

    Note that here I remove any interaction that /w genes that are not found
    in the dataset. Note that this is not necessarily the case in liana-r.
    There, I assign the expression of those with missing subunits to 0, while
    those without any subunit present are implicitly filtered.
    """

    # Remove those without any subunit
    resource = resource[(np.isin(resource.ligand, var_names)) &
                        (np.isin(resource.receptor, var_names))]

    # Only keep interactions /w complexes for which all subunits are present
    missing_comps = resource[resource.interaction.str.contains('_')].copy()
    missing_comps['all.units'] = \
        missing_comps['ligand.complex'] + '_' + missing_comps[
            'receptor.complex']

    # Get those not with all subunits
    missing_comps = missing_comps[np.logical_not(
        [all([x in var_names for x in entity.split('_')])
         for entity in missing_comps['all.units']]
    )]
    # Filter them
    return resource[~resource.interaction.isin(missing_comps.interaction)]
