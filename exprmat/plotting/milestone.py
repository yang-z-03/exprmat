
from typing_extensions import Literal
import igraph
import numpy as np
import pandas as pd
from typing import Union, Optional, Iterable
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scanpy as sc


def milestones(
    adata,
    basis: Union[None, str] = 'umap',
    trajectory_key = 'ppt',
    annotate: bool = False,
    title = None,
    subset: Optional[Iterable] = None,
    ax = None,
    sort = False,
    cmap = 'turbo',
    **kwargs,
):
    
    emb = adata.obsm[basis]
    kwargs['sort_order'] = sort

    if not sort: order = adata.obs[f'{trajectory_key}.pseudotime'].sort_values().index
    else: order = adata.obs[f'{trajectory_key}.pseudotime'].sort_values().index
    order = order[order.isin(subset)] if subset is not None else order

    if "color" in kwargs:
        kwargs.pop("color")
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if "edgecolor" not in kwargs: kwargs["edgecolor"] = "none"
    if "s" not in kwargs: kwargs["s"] = 120000 / adata.shape[0]

    ax.scatter(
        adata[order].obsm[basis][:, 0],
        adata[order].obsm[basis][:, 1],
        c = gen_milestones_gradients(adata, key = trajectory_key, palette = cmap)[order].values,
        marker = ".",
        rasterized = True,
        plotnonfinite = True,
        ** kwargs,
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.autoscale_view()

    if annotate:
        
        R = adata.obsm[trajectory_key]
        proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

        X = proj[list(adata.uns[f"{trajectory_key}.graph"]["milestones"].values()), :]
        adata_m = sc.AnnData(
            X, dtype = X.dtype,
            obs = dict(mil = list(adata.uns[f"{trajectory_key}.graph"]["milestones"].keys())),
            obsm = {basis: X},
        )

        adata_m.obs["mil"] = adata_m.obs["mil"].astype("category")

        sc.pl.embedding(
            adata_m,
            basis,
            color = "mil",
            title = title,
            legend_loc = "on data",
            ax = ax,
            alpha = 0,
            legend_fontoutline = True,
            show = False,
        )


def gen_milestones_gradients(adata, palette = 'turbo', key = 'ppt', seg_order = None):

    seg_order = adata.obs[f'{key}.segs'].unique() \
        if seg_order is None else seg_order

    if "milestones.colors" not in adata.uns or len(adata.uns["milestones.colors"]) == 1:
        from exprmat.plotting.palettes import get_palette
        adata.uns['milestones.colors'] = get_palette(palette, len(seg_order))

    def milestones_prog(s):
        
        cfrom = adata.obs.t[adata.obs.seg == s].idxmin()
        cto = adata.obs.t[adata.obs.seg == s].idxmax()
        mfrom = adata.obs.milestones[cfrom]
        mto = adata.obs.milestones[cto]
        mfrom_c = adata.uns["milestones_colors"][
            np.argwhere(adata.obs.milestones.cat.categories == mfrom)[0][0]
        ]
        mto_c = adata.uns["milestones_colors"][
            np.argwhere(adata.obs.milestones.cat.categories == mto)[0][0]
        ]

        from matplotlib.colors import LinearSegmentedColormap, to_hex
        cm = LinearSegmentedColormap.from_list("test", [mfrom_c, mto_c], N=1000)
        pst = (
            adata.obs.t[adata.obs.seg == s] - adata.obs.t[adata.obs.seg == s].min()
        ) / (
            adata.obs.t[adata.obs.seg == s].max()
            - adata.obs.t[adata.obs.seg == s].min()
        )
        return pd.Series(list(map(to_hex, cm(pst))), index = pst.index)

    return pd.concat(list(map(milestones_prog, seg_order)))

