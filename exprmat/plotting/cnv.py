
import anndata as ad
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
from matplotlib.colors import Colormap, TwoSlopeNorm
from exprmat.ansi import error


def chromosome_heatmap(
    adata, *,
    groupby: str = "leiden.cnv",
    use_rep: str = "cnv",
    cmap = "bwr",
    figsize: tuple[int, int] = (16, 10), 
    draw_autosome_only = False,
    sample_name = None, **kwargs
):
    """
    Plot a heatmap of smoothed gene expression by chromosome.

    Parameters
    ----------
    adata
        Annotated data matrix

    groupby
        group the cells by a categorical variable from adata.obs. It usually makes
        sense to either group by unsupervised clustering obtained from or a cell-type label.

    use_rep
        Key under which the result from `infercnv` are stored.
        
    **kwargs
        Arguments passed on to :func:`scanpy.pl.heatmap`.

    """

    if groupby not in adata.obs.columns:
        error(f"'{groupby}' is not in `adata.obs`.")

    tmp_adata = ad.AnnData(X = adata.obsm[use_rep], obs = adata.obs)
    tmp_adata.uns[use_rep] = adata.uns[use_rep]

    # re-sort, as saving & loading anndata destroys the order
    chr_pos_dict = dict(sorted(adata.uns[use_rep]["chr.pos"].items(), key = lambda x: x[1]))
    chr_pos = list(chr_pos_dict.values())

    # center color map at 0
    tmp_data = tmp_adata.X.data if issparse(tmp_adata.X) else tmp_adata.X
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if vmin is None: vmin = np.nanmin(tmp_data)
    if vmax is None: vmax = np.nanmax(tmp_data)
    kwargs["norm"] = TwoSlopeNorm(0, vmin = vmin, vmax = vmax)

    # add chromosome annotations
    var_group_positions = list(zip(chr_pos, chr_pos[1:] + [tmp_adata.shape[1]], strict=False))

    labels = list(chr_pos_dict.keys())
    if draw_autosome_only:
        labels = [x if len(x) <= 2 else '' for x in labels]

    return_ax_dic = sc.pl.heatmap(
        tmp_adata,
        var_names = tmp_adata.var.index.values,
        groupby = groupby,
        figsize = figsize,
        cmap = cmap,
        show_gene_labels = False,
        var_group_positions = var_group_positions,
        var_group_labels = list(chr_pos_dict.keys()),
        show = False,
        ** kwargs,
    )

    return_ax_dic["heatmap_ax"].vlines(
        chr_pos[1:], 
        lw = 0.6, 
        ymin = 0, 
        ymax = tmp_adata.shape[0]
    )

    return return_ax_dic["heatmap_ax"].figure


def chromosome_heatmap_summary(
    adata: ad.AnnData, *,
    groupby: str = "leiden.cnv",
    use_rep: str = "cnv",
    cmap: str | Colormap = "bwr",
    figsize: tuple[int, int] = (16, 10),**kwargs,
):
    """
    Plot a heatmap of average of the smoothed gene expression by chromosome per category in groupby.
    """

    if groupby not in adata.obs.columns:
        error(f"'{groupby}' is not in `adata.obs`.")

    # TODO this dirty hack repeats each row 10 times, since scanpy heatmap cannot 
    # really handle it if there's just one observation per row. Scanpy matrixplot 
    # is not an option, since it plots each gene individually.

    import pandas as pd
    groups = adata.obs[groupby].unique()
    tmp_obs = pd.DataFrame()
    tmp_obs[groupby] = np.hstack([np.repeat(x, 10) for x in groups])
    tmp_obs.index = tmp_obs.index.astype(str)

    def _get_group_mean(group):
        group_mean = np.mean(adata.obsm[f"X_{use_rep}"][adata.obs[groupby].values == group, :], axis=0)
        if len(group_mean.shape) == 1:
            # derived from an array instead of sparse matrix -> 1 dim instead of 2
            group_mean = group_mean[np.newaxis, :]
        return group_mean

    tmp_adata = sc.AnnData(
        X = np.vstack([np.repeat(_get_group_mean(group), 10, axis = 0) for group in groups]), 
        obs = tmp_obs
    )

    tmp_adata.uns[use_rep] = adata.uns[use_rep]
    chr_pos_dict = dict(sorted(adata.uns[use_rep]["chr.pos"].items(), key = lambda x: x[1]))
    chr_pos = list(chr_pos_dict.values())

    # center color map at 0
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if vmin is None: vmin = np.min(tmp_adata.X)
    if vmax is None: vmax = np.max(tmp_adata.X)
    kwargs["norm"] = TwoSlopeNorm(0, vmin=vmin, vmax=vmax)

    # add chromosome annotations
    var_group_positions = list(zip(chr_pos, chr_pos[1:] + [tmp_adata.shape[1]], strict = False))

    return_ax_dic = sc.pl.heatmap(
        tmp_adata,
        var_names = tmp_adata.var.index.values,
        groupby = groupby,
        figsize = figsize,
        cmap = cmap,
        show_gene_labels = False,
        var_group_positions = var_group_positions,
        var_group_labels = list(chr_pos_dict.keys()),
        show = False, **kwargs
    )

    return_ax_dic["heatmap_ax"].vlines(
        chr_pos[1:], 
        lw = 0.6, 
        ymin = -1, 
        ymax = tmp_adata.shape[0]
    )

    return return_ax_dic