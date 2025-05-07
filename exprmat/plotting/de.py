
import functools
import operator
from collections.abc import Mapping, Sequence
from copy import copy
import pandas as pd
import numpy as np

from exprmat.ansi import warning, error


def get_marker_df(  # noqa: PLR0912
    adata, group, *,
    key: str = "markers",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
    gene_symbols: str | None = None,
):
    
    if isinstance(group, str): group = [group]
    if group is None: group = list(adata.uns[key]['names'].dtype.names)
    method = adata.uns[key]["params"]["method"]
    if method == "logreg": colnames = ["names", "scores"]
    else: colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]

    d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
    d = pd.concat(d, axis = 1, names = [None, "group"], keys = colnames)
    d = d.stack(level = 1, future_stack = True).reset_index()
    d["group"] = pd.Categorical(d["group"], categories = group)
    d = d.sort_values(["group", "level_0"]).drop(columns = "level_0")

    if method != "logreg":
        if pval_cutoff is not None:
            d = d[d["pvals_adj"] < pval_cutoff]
        if log2fc_min is not None:
            d = d[d["logfoldchanges"] > log2fc_min]
        if log2fc_max is not None:
            d = d[d["logfoldchanges"] < log2fc_max]

    if gene_symbols is not None:
        d = d.join(adata.var[gene_symbols], on = "names")

    for pts, name in {"pts": "pct_nz_group", "pts_rest": "pct_nz_reference"}.items():
        if pts in adata.uns[key]:
            pts_df = (
                adata.uns[key][pts][group]
                .rename_axis(index = "names")
                .reset_index()
                .melt(id_vars = "names", var_name = "group", value_name = name)
            )
            d = d.merge(pts_df)

    # remove group column for backward compat if len(group) == 1
    if len(group) == 1: d.drop(columns = "group", inplace = True)
    d = d.rename(columns = {
        'logfoldchanges': 'lfc', 
        'pvals': 'p', 
        'pvals_adj': 'q',
        'pct_nz_reference': 'pct.reference',
        'pct_nz_group': 'pct'
    })
    d['log10.p'] = -1 * np.log10(d['p'])
    d['log10.q'] = -1 * np.log10(d['q'])
    return d.reset_index(drop = True)


def filter_value_df(
    df, method,
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
    pct_min = 0.25,
    pct_reference_max = 0.50
):
    if method != "logreg":
        if pval_cutoff is not None:
            df = df[df["q"] < pval_cutoff]
        if log2fc_min is not None:
            df = df[df["lfc"] > log2fc_min]
        if log2fc_max is not None:
            df = df[df["lfc"] < log2fc_max]
    
    if "pct" in df.columns: df = df[df["pct"] > pct_min]
    if "pct.reference" in df.columns: df = df[df["pct.reference"] < pct_reference_max]

    return df


def get_plotting_values(
    adata, values_to_plot,
    gene_names: Sequence[str],
    *, groups: Sequence[str] | None = None,
    key: str | None = "markers",
    gene_symbols: str | None = None,
):
    
    valid_options = ["scores", "lfc", "p", "q", "log10.p", "log10.q"]
    if values_to_plot not in valid_options:
        error(f"given value_to_plot: '{values_to_plot}' is not valid.")

    values_df = None
    check_done = False
    
    if groups is None:
        groups = adata.uns[key]["names"].dtype.names
    
    if values_to_plot is not None:
        df_list = []
        for group in groups:
            df = adata.uns[key]['differential'][group]
            if gene_symbols is not None:
                df["names"] = df[gene_symbols]

            # check that all genes are present in the df as sc.tl.rank_genes_groups
            # can be called with only top genes
            if not check_done and df.shape[0] < adata.shape[1]:
                error(f"`markers()` should run with all genes. currently only {df.shape[0]} genes are found.")
            df["group"] = group
            df_list.append(df)

        values_df = pd.concat(df_list)
        if values_to_plot.startswith("log10"):
            column = values_to_plot.replace("log10.", "")
        else: column = values_to_plot
        values_df = pd.pivot(
            values_df, index = "names", columns = "group", values=column
        ).fillna(1)

        if values_to_plot in ["log10.p", "log10.q"]:
            values_df = -1 * np.log10(values_df)

        values_df = values_df.loc[gene_names].T

    return values_df


def marker_plot(
    adata, plot_type: str = "dotplot", *,
    groups: str | Sequence[str] | None = None,
    n_genes: int | None = None,
    groupby: str | None = None,
    values_to_plot: str | None = None,
    var_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    min_logfoldchange: float | None = 1,
    max_logfoldchange = None,
    max_p_adjust = 0.05,
    min_pct = 0.25,
    max_pct_reference = 0.75,
    key: str | None = None,
    show: bool | None = None,
    save: bool | None = None,
    gene_symbols: str | None = None,
    sample_name = None,
    **kwds,
):
    
    if var_names is not None and n_genes is not None:
        error("the arguments n_genes and var_names are mutually exclusive.")

    if key is None: key = "markers"
    if groupby is None: groupby = str(adata.uns[key]["params"]["groupby"])
    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups

    if var_names is not None:
        
        if isinstance(var_names, Mapping):
            # get a single list of all gene names in the dictionary
            var_names_list = functools.reduce(
                operator.iadd, [list(x) for x in var_names.values()], []
            )

        elif isinstance(var_names, str):
            var_names_list = [var_names]
        else: var_names_list = var_names
    
    else:
        # set n_genes = 5 as default when none of the options is given
        if n_genes is None: n_genes = 5

        # dict in which each group is the key and the n_genes are the values
        var_names = {}
        var_names_list = []
        for group in group_names:
            
            df = adata.uns[key]['differential'][group]
            df = filter_value_df(
                df, method = adata.uns[key]['params']['method'],
                log2fc_min = min_logfoldchange,
                log2fc_max = max_logfoldchange,
                pct_min = min_pct,
                pct_reference_max = max_pct_reference,
                pval_cutoff = max_p_adjust
            )

            if gene_symbols is not None:
                df["names"] = df[gene_symbols]

            genes_list = df.names[df.names.notnull()].tolist()

            if len(genes_list) == 0:
                warning(f"no genes found for group {group}")
                continue

            genes_list = genes_list[n_genes:] if n_genes < 0 else genes_list[:n_genes]
            var_names[group] = genes_list
            var_names_list.extend(genes_list)


    if plot_type in ["dotplot", "matrixplot"]:

        # these two types of plots can also show score, logfoldchange and pvalues, 
        # in general any value from rank genes groups

        title = None
        values_df = None
        if values_to_plot is not None:
            values_df = get_plotting_values(
                adata,
                values_to_plot,
                var_names_list,
                key = key,
                gene_symbols = gene_symbols,
            )
            title = values_to_plot
            if values_to_plot == "lfc": title = "Log fold change"
            else: title = values_to_plot.replace(".", " ")

        if plot_type == "dotplot":

            from scanpy.plotting._dotplot import dotplot
            _pl = dotplot(
                adata,
                var_names,
                groupby,
                dot_color_df = values_df,
                return_fig = True,
                gene_symbols = gene_symbols,
                **kwds
            )

            if title is not None and "colorbar_title" not in kwds:
                _pl.legend(colorbar_title = title)

        elif plot_type == "matrixplot":

            from scanpy.plotting._matrixplot import matrixplot
            _pl = matrixplot(
                adata,
                var_names,
                groupby,
                values_df = values_df,
                return_fig = True,
                gene_symbols = gene_symbols,
                **kwds,
            )

            if title is not None and "colorbar_title" not in kwds:
                _pl.legend(title = title)

        return _pl

    elif plot_type == "stacked_violin":

        from scanpy.plotting._stacked_violin import stacked_violin
        _pl = stacked_violin(
            adata,
            var_names,
            groupby,
            return_fig = True,
            gene_symbols = gene_symbols,
            **kwds,
        )
        return _pl
    
    elif plot_type == "heatmap":

        from scanpy.plotting._anndata import heatmap
        return heatmap(
            adata,
            var_names,
            groupby,
            show=show,
            save=save,
            gene_symbols = gene_symbols,
            **kwds,
        )

    elif plot_type == "tracksplot":

        from scanpy.plotting._anndata import tracksplot
        return tracksplot(
            adata,
            var_names,
            groupby,
            show=show,
            save=save,
            gene_symbols = gene_symbols,
            **kwds,
        )