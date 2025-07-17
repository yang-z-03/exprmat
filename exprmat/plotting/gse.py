
import sys
import operator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.category import UnitData
from matplotlib.lines import Line2D

from exprmat.ansi import error, info


class normalize_midpoint(Normalize):
    
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip = None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y, left = -np.inf, right = np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left = -np.inf, right = np.inf)


def zscore(data2d: pd.DataFrame, axis: Optional[int] = 0):
    
    if axis is None:
        # normalized to mean and std using entire matrix
        # z_scored = (data2d - data2d.values.mean()) / data2d.values.std(ddof=1)
        return data2d
    
    assert axis in [0, 1]
    z_scored = data2d.apply(
        lambda x: (x - x.mean()) / x.std(ddof=1), axis = operator.xor(1, axis)
    )

    return z_scored


class nesplot:

    def __init__(
        self,
        term: str,
        tag: Sequence[int],
        runes: Sequence[float],
        nes: float,
        pval: float,
        fdr: float,
        rank_metric: Optional[Sequence[float]] = None,
        pheno_pos: str = "",
        pheno_neg: str = "",
        color: Optional[str] = "#88C544",
        figsize: Tuple[float, float] = (4, 4),
        cmap: str = "seismic",
        ax: Optional[plt.Axes] = None,
        module = 'gsea',
        title = None,
        ** kwargs
    ):
        
        self._x = np.arange(len(runes))
        self.rankings = None
        self._zero_score_ind = None
        self._z_score_label = None
        self.color = color
        self.module = module

        if rank_metric is not None:
            self.rankings = np.asarray(rank_metric)
            self._zero_score_ind = np.abs(self.rankings).argmin()
            self._z_score_label = str(self._zero_score_ind)

        self.running_es = np.asarray(runes)

        self.figsize = figsize
        self.term = term
        self.cmap = cmap

        self._pos_label = pheno_pos
        self._neg_label = pheno_neg
        self._hit_indices = tag
        
        if self.module == "ssgsea":
            self._nes_label  = "Enrichment Score: " + "{:.3f}".format(float(nes))
            self._fdr_label  = "False discovery rate: -"
            self._pval_label = "p value: -"
        
        else:
            self._nes_label  = "Normalized enrichment score: " + "{:.3f}".format(float(nes))
            self._fdr_label  = "False discovery rate: " + "{:.3f}".format(float(fdr))
            self._pval_label = "p value: " + "{:.3f}".format(float(pval))

        if ax is None:
            if hasattr(sys, "ps1"):
                # working inside python console, show figure
                self.fig = plt.figure(figsize=self.figsize, facecolor = "white")
            else:
                # if working on command line, don't show figure
                self.fig = Figure(figsize=self.figsize, facecolor = "white")
                self._canvas = FigureCanvas(self.fig)

        elif isinstance(ax, plt.Axes): self.fig = ax.figure
        else: error("`ax` must be matplotlib axes.")
        self.fig.suptitle(
            self.term if title is None else title, fontsize = 11, 
            wrap = True, fontweight = "bold"
        )


    def axes_rank(self, rect):
        
        ax1 = self.fig.add_axes(rect)
        if self.module == "ssgsea":
            ax1.fill_between(self._x, y1 = np.log(self.rankings), y2=0, color = "#C9D3DB")
            ax1.set_ylabel("Ranked metric (log)", fontsize = 9)
        else:
            ax1.fill_between(self._x, y1=self.rankings, y2 = 0, color="#C9D3DB")
            ax1.set_ylabel("Ranked metric", fontsize = 9)

        ax1.text(
            0.05, 0.9,
            self._pos_label,
            color = "red",
            horizontalalignment = "left",
            verticalalignment = "top",
            transform = ax1.transAxes,
        )

        ax1.text(
            0.95, 0.05,
            self._neg_label,
            color = "Blue",
            horizontalalignment = "right",
            verticalalignment = "bottom",
            transform = ax1.transAxes,
        )

        # the x coords of this transformation are data, and the y coord are axes
        trans1 = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.vlines(
            self._zero_score_ind, 0, 1, linewidth = 0.5,
            transform = trans1, linestyles = "--", color = "grey",
        )

        hap = self._zero_score_ind / max(self._x)
        if hap < 0.25: ha = "left"
        elif hap > 0.75: ha = "right"
        else: ha = "center"
        ax1.text(
            hap, 0.5, self._z_score_label,
            horizontalalignment = ha, verticalalignment = "center",
            transform = ax1.transAxes, fontsize = 9,
        )

        ax1.set_xlabel("Gene ranking", fontsize = 9)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(
            axis = "both", which = "both", 
            top = False, right = False, left = True, labelsize = 9
        )
        ax1.locator_params(axis = "y", nbins = 5)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc, tick_num: "{:.0f}".format(tick_loc))
        )


    def axes_hits(self, rect, bottom: bool = False):
        
        ax2 = self.fig.add_axes(rect)
        # the x coords of this transformation are data, and the y coord are axes
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        # to make axes shared with same x cooridincates, make the vlines same ranges to x
        ax2.vlines(
            [self._x[0], self._x[-1]],
            0, 1, linewidth = 0.5, transform = trans2,
            color = "white", alpha = 0,
        )

        # add hits line
        ax2.vlines(self._hit_indices, 0, 1, linewidth = 0.5, transform = trans2, color = "black")

        ax2.tick_params(
            axis = "both", which = "both",
            bottom = bottom, top = False, right = False, left = False,
            labelbottom = bottom, labelleft = False,
        )

        if bottom:
            ax2.set_xlabel("Gene ranking", fontsize = 9)
            ax2.spines["bottom"].set_visible(True)


    def axes_cmap(self, rect):
        
        # center color map at midpoint = 0
        mat = self.rankings
        if self.rankings is None:
            mat = self.running_es
        vmin = np.percentile(mat.min(), 2)
        vmax = np.percentile(mat.max(), 98)
        midnorm = normalize_midpoint(vmin = vmin, vcenter = 0, vmax = vmax)
        
        # colormap
        ax3 = self.fig.add_axes(rect)
        ax3.pcolormesh(
            mat[np.newaxis, :],
            rasterized = True,
            norm = midnorm,
            cmap = self.cmap,
        )

        ax3.spines["bottom"].set_visible(False)
        ax3.tick_params(
            axis = "both", which = "both",
            bottom = False, top = False, right = False, left = False,
            labelbottom = False, labelleft = False,
        )


    def axes_stat(self, rect):

        # enrichment score plot
        ax4 = self.fig.add_axes(rect)
        ax4.plot(self._x, self.running_es, linewidth = 4, color = self.color)

        if '-' in self._nes_label:
            ax4.text(0.025, 0.05, self._fdr_label, transform = ax4.transAxes, fontsize = 9)
            ax4.text(0.025, 0.15, self._pval_label, transform = ax4.transAxes, fontsize = 9)
            ax4.text(0.025, 0.25, self._nes_label, transform = ax4.transAxes, fontsize = 9)
        else:
            ax4.text(0.975, 0.95, self._fdr_label, ha = 'right', va = 'top', transform = ax4.transAxes, fontsize = 9)
            ax4.text(0.975, 0.85, self._pval_label, ha = 'right', va = 'top', transform = ax4.transAxes, fontsize = 9)
            ax4.text(0.975, 0.75, self._nes_label, ha = 'right', va = 'top', transform = ax4.transAxes, fontsize = 9)

        # the y coords of this transformation are data, and the x coord are axes
        trans4 = transforms.blended_transform_factory(ax4.transAxes, ax4.transData)
        ax4.hlines(0, 0, 1, linewidth = 1, transform = trans4, color = "grey")
        ax4.set_ylabel("Enrichment score", fontsize = 9)

        ax4.tick_params(
            axis = "both", which = "both",
            bottom = False, top = False, right = False, labelbottom = False,
            labelsize = 9
        )
        ax4.locator_params(axis = "y", nbins = 5)
        ax4.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc, tick_num: "{:.1f}".format(tick_loc))
        )
        self.ax = ax4


    def add_axes(self):

        left = 0.1
        width = 0.8
        bottom = 0.1
        height = 0

        stat_height_ratio = 0.4
        hits_height_ratio = 0.05
        cmap_height_ratio = 0.05
        rank_height_ratio = 0.3
        
        if self.rankings is None:
            rank_height_ratio = 0
            cmap_height_ratio = 0
        
        base = 0.8 / (
            stat_height_ratio
            + hits_height_ratio
            + cmap_height_ratio
            + rank_height_ratio
        )

        if self.rankings is not None:
            height = rank_height_ratio * base
            self.axes_rank([left, bottom, width, height])
            bottom += height
            height = cmap_height_ratio * base
            self.axes_cmap([left, bottom, width, height])
            bottom += height

        height = hits_height_ratio * base
        self.axes_hits(
            [left, bottom, width, height], bottom=False if bottom > 0.1 else True
        )

        bottom += height
        height = stat_height_ratio * base
        self.axes_stat([left, bottom, width, height])


def esplot(
    adata, *, gsea, terms, figsize = (4, 4), title = None,
    colors = None, show_ranking = True, sample_name = None
):
    
    gsdict = adata.uns[gsea]['results'][terms]
    ranking = pd.DataFrame({
        'gene_names': adata.uns[gsea]['config']['ranking.key'],
        'value': adata.uns[gsea]['config']['ranking.value']
    })

    ranking.set_index('gene_names')
    g = nesplot(
        term = terms,
        tag = gsdict["hits"],
        rank_metric = ranking['value'] if (show_ranking) else None,
        runes = gsdict["res"],
        nes = gsdict["nes"],
        pval = gsdict["p"],
        fdr = gsdict["fdr"],
        pheno_pos = '', # adata.uns[gsea]['config']['positive'],
        pheno_neg = '', # adata.uns[gsea]['config']['negative'],
        color = colors,
        title = title,
        figsize = figsize,
    )

    g.add_axes()
    return g.fig


class heatmapplot(object):

    def __init__(
        self, df: pd.DataFrame,
        z_score: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5, 5),
        cmap: Optional[str] = 'turbo',
        xticklabels: bool = True,
        yticklabels: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ):
        self.title = "" if title is None else title
        self.figsize = figsize
        self.xticklabels = xticklabels
        self.yticklabels = yticklabels
        self.ax = ax

        # scale dataframe
        df = df.astype(float)
        df = zscore(df, axis = z_score)
        df = df.iloc[::-1]
        self.data = df
        self.cbar_title = "lognorm" if z_score is None else "scaled"
        self.cmap = cmap
        self._zscore = z_score


    def _skip_ticks(self, labels, tickevery):

        n = len(labels)
        if tickevery == 0:
            ticks, labels = [], []
        elif tickevery == 1:
            ticks, labels = np.arange(n) + 0.5, labels
        else:
            start, end, step = 0, n, tickevery
            ticks = np.arange(start, end, step) + 0.5
            labels = labels[start:end:step]
        return ticks, labels


    def _auto_ticks(self, ax, labels, axis):

        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        (tick,) = ax.xaxis.set_ticks([0])
        fontsize = tick.label1.get_size()
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1: tickevery = 1
        else: tickevery = len(labels) // max_ticks + 1
        return tickevery


    def get_ax(self):

        if (self.ax is not None) and isinstance(self.ax, plt.Axes):
            self.fig = self.ax.figure
            return self.ax
        
        elif hasattr(sys, "ps1"):
            fig = plt.figure(figsize = self.figsize)
        
        else:
            fig = Figure(figsize = self.figsize)
            canvas = FigureCanvas(fig)
        
        ax = fig.add_subplot(111)
        self.fig = fig
        return ax


    def draw(self):

        df = self.data
        ax = self.get_ax()
        vmin = np.percentile(df, 2)
        vmax = np.percentile(df, 98)

        from matplotlib.ticker import MaxNLocator
        if self._zscore is None:
            norm = Normalize(vmin = vmin, vmax = vmax)
            cbar_locator = MaxNLocator(nbins = 5, integer = True)
        else:
            norm = normalize_midpoint(vmin=vmin, vmax = vmax, vcenter = 0)
            cbar_locator = MaxNLocator(nbins=3, symmetric = True)
        
        matrix = ax.pcolormesh(
            df.values,
            cmap = self.cmap,
            norm = norm,
            rasterized=True,
        )

        xstep = self._auto_ticks(ax, df.columns.values, 0)
        ystep = self._auto_ticks(ax, df.index.values, 1)
        xticks, xlabels = self._skip_ticks(df.columns.values, tickevery = xstep)
        yticks, ylabels = self._skip_ticks(df.index.values, tickevery = ystep)
        ax.set_ylim([0, len(df)])
        ax.set(xticks = xticks, yticks = yticks)
        ax.set_xticklabels(
            xlabels if self.xticklabels else "", fontsize=14, rotation=90
        )
        ax.set_yticklabels(ylabels if self.yticklabels else "", fontsize = 9)
        ax.set_title(self.title, fontsize = 11, fontweight = "bold")
        ax.tick_params(
            axis = "both", which = "both", 
            bottom = False, top = False, right = False, left = False
        )
        
        cbar = self.fig.colorbar(matrix, shrink = 0.3, aspect = 10)
        cbar.ax.yaxis.set_tick_params(
            color = "white", direction = "in", left = True, right = True
        )

        # colorbar, make sure to specify tick locations to match desired ticklabels
        cbar.locator = cbar_locator
        cbar.update_ticks()
        cbar.ax.set_title(self.cbar_title, loc = "left")
        for key, spine in cbar.ax.spines.items():
            spine.set_visible(False)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

        return ax
    

def heatmap(
    df: pd.DataFrame,
    z_score: Optional[int] = None,
    title: str = "",
    figsize: Tuple[float, float] = (4, 4),
    cmap: Optional[str] = 'turbo',
    xticklabels: bool = True,
    yticklabels: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    
    ht = heatmapplot(
        df = df,
        z_score = z_score,
        title = title,
        figsize = figsize,
        cmap = cmap,
        xticklabels = xticklabels,
        yticklabels = yticklabels,
        ax = ax, **kwargs
    )

    ax = ht.draw()
    return ax.figure


class dotplot(object):
    
    def __init__(
        self,
        df: pd.DataFrame,
        x: Optional[str] = None,
        y: str = "term",
        hue: str = "q",
        dot_scale: float = 5.0,
        x_order: Optional[List[str]] = None,
        y_order: Optional[List[str]] = None,
        thresh: float = 0.05,
        n_terms: int = 10,
        terms = None,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (6, 5.5),
        cmap: str = 'turbo',
        marker: str = "o",
        formatter = lambda x: x,
        ** kwargs,
    ):
        
        self.marker = marker
        self.y = y
        self.x = x
        self.x_order = x_order
        self.y_order = y_order
        self.hue = str(hue)
        self.colname = str(hue)
        self.figsize = figsize
        self.cmap = cmap
        self.ax = ax
        self.scale = dot_scale
        self.title = title
        self.n_terms = n_terms
        self.terms = terms
        self.thresh = thresh
        self.formatter = formatter
        self.data = self.process(df)


    def isfloat(self, x):
        try: float(x)
        except: return False
        else: return True


    def process(self, df: pd.DataFrame):
        
        # check if any values in `df[colname]` can't be coerced to floats
        can_be_coerced = df[self.colname].map(self.isfloat).sum()
        if can_be_coerced < df.shape[0]:
            error("some value in %s could not be typecast to `float`" % self.colname)
        # subset
        mask = df[self.colname] <= self.thresh
        if self.colname in ["score", "nes", "es", "or"]: mask.loc[:] = True

        df = df.loc[mask]
        if df.shape[0] < 1:
            error("no enriched terms when cutoff = %s" % self.thresh)
        
        self.cbar_title = self.colname
        
        colnd = {
            "fdr": "q",
            "p": "p"
        }

        # impute the 0s in pval, fdr for visualization purpose
        if self.colname in colnd.keys():
            # if all values are zeros, raise error
            if not any(df[self.colname].abs() > 0):
                error(f"can not detetermine colormap. All values in {self.colname} are 0s")
            df = df.sort_values(by = self.colname)
            df[self.colname] = df[self.colname] + np.finfo(float).eps
            df = df.assign(p_inv = np.log10(1 / df[self.colname].astype(float)))
            _t = colnd[self.colname]

            self.cbar_title = f'-log({self.colname})'
            self.colname = "p_inv"

        if self.terms is not None:
            if 'term' in df.columns:
                df = df.loc[[x in self.terms for x in df['term']], :].copy()
            elif 'name' in df.columns:
                df = df.loc[[x in self.terms for x in df['name']], :].copy()
            df = df.sort_values(by = self.colname)
        
        # get top terms; sort ascending
        elif ((self.x is not None)
            and (self.x in df.columns)
            and (not all(df[self.x].map(self.isfloat)))
        ):

            df = ( # if x is numeric column, get top term of each group
                df.groupby(self.x)
                  .apply(lambda _x: _x.sort_values(by = self.colname).tail(self.n_terms))
                  .reset_index(drop=True)
            )

        else: df = df.sort_values(by = self.colname).tail(self.n_terms)  # acending
        
        # get scatter area
        if df.columns.isin(["overlap", "tag"]).any():
            ol = df.columns[df.columns.isin(["overlap", "tag"])]
            temp = (
                df[ol].squeeze(axis = 1).str.split("/", expand=True).astype(int)
            )  # axis = 1, in case you have only 1 row
            df = df.assign(hits_ratio = temp.iloc[:, 0] / temp.iloc[:, 1])
        else: df = df.assign(hits_ratio = 1.0)  # if overlap column missing
        return df


    def hierarchical_clust(self, mat, method, metric) -> List[int]:
        # mat.shape -> [n_sample, m_features]
        Y0 = sch.linkage(mat, method = method, metric = metric)
        Z0 = sch.dendrogram(
            Y0, orientation = "left", no_plot = True,
            distance_sort = "descending",
        )
        idx = Z0["leaves"][::-1]  # reverse the order to make the view better
        return idx


    def get_x_order(
        self, method: str = "single", metric: str = "euclidean"
    ) -> List[str]:
        
        if hasattr(self.x_order, "__len__"):
            return self.x_order
        mat = self.data.pivot(
            index = self.y,
            columns = self.x,
            values = self.colname,  # [self.colname, "hits_ratio"],
        ).fillna(0)
        idx = self.hierarchical_clust(mat.T, method, metric)
        return list(mat.columns[idx])


    def get_y_order(
        self, method: str = "single", metric: str = "euclidean"
    ) -> List[str]:
        
        if hasattr(self.y_order, "__len__"):
            return self.y_order
        mat = self.data.pivot(
            index = self.y,
            columns = self.x,
            values = self.colname,
        ).fillna(0)
        idx = self.hierarchical_clust(mat, method, metric)
        ys = list(mat.index[idx])
        return ys


    def get_ax(self):
        
        if (self.ax is not None) and isinstance(self.ax, plt.Axes):
            self.fig = self.ax.figure
            return self.ax
        
        elif hasattr(sys, "ps1"): fig = plt.figure(figsize=self.figsize)
        else:
            fig = Figure(figsize = self.figsize)
            canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)
        self.fig = fig
        return ax


    def set_x(self):
        
        x = self.x
        xlabel = ""
        
        if (x is not None) and (x in self.data.columns):
            if x == 'score': xlabel = 'Combined score'
            elif x == 'or': xlabel = 'Odds ratio'
            elif x == 'nes': xlabel = 'Normalized enrichment ratio'
            else: xlabel = x
        elif "score" in self.data.columns:
            xlabel = "Combined score"
            x = 'score'
        elif "or" in self.data.columns:
            xlabel = "Odds ratio"
            x = 'or'
        elif "nes" in self.data.columns:
            xlabel = "Normalized ES"
            x = 'nes'
        
        else:
            # revert back to p_inv
            x = self.colname
            xlabel = self.cbar_title

        return x, xlabel


    def scatter(
        self,
        outer_ring: bool = False,
    ):
        
        df = self.data.assign(
            area = (self.data["hits_ratio"] * self.scale * plt.rcParams["lines.markersize"]).pow(2)
        )

        # filtering and renaming
        y = self.y
        df[y] = [self.formatter(z) for z in df[y]]
        df = df.loc[~ df[y].isna(), :].copy()

        info(f'retreived {len(df)} terms for plotting.')

        colmap = df[self.colname].astype(int)
        vmin = np.percentile(colmap.min(), 2)
        vmax = np.percentile(colmap.max(), 98)
        ax = self.get_ax()
        x, xlabel = self.set_x()

        # if x axis is numberic, prettifiy the plot with the numberic order
        if all(df[x].map(self.isfloat)): df = df.sort_values(by = x)
        
        # set x, y order if set
        xunits = UnitData(self.get_x_order()) if self.x_order else None
        yunits = UnitData(self.get_y_order()) if self.y_order else None

        # outer ring
        if outer_ring:
            smax = df["area"].max()
            
            blk_sc = ax.scatter(
                x = x, y = y, s = smax * 1.6,
                edgecolors = "none", c = "black",
                data = df, marker = self.marker,
                xunits = xunits,  # set x categorical order
                yunits = yunits,  # set y categorical order
                zorder = 0,
            )

            wht_sc = ax.scatter(
                x = x, y = y, s = smax * 1.3,
                edgecolors = "none", c = "white",
                data = df, marker = self.marker,
                xunits = xunits,  # set x categorical order
                yunits = yunits,  # set y categorical order
                zorder = 1,
            )
            
        # inner circle

        # df2 = df.copy()
        # if self.formatter is not None:
        #     df2[y] = [self.formatter(z) for z in df2[y]]
        
        sc = ax.scatter(
            x = x, y = y, data = df,
            s = "area", edgecolors = "none",
            c = self.colname, cmap = self.cmap,
            vmin = vmin,
            vmax = vmax,
            marker = self.marker,
            xunits = xunits,  # set x categorical order
            yunits = yunits,  # set y categorical order
            zorder = 2,
        )

        ax.set_xlabel(xlabel, fontsize = 9)
        ax.xaxis.set_tick_params(labelsize = 9)
        ax.yaxis.set_tick_params(labelsize = 9)
        ax.set_axisbelow(True)  # set grid blew other element
        ax.grid(axis = "y", zorder = -1)  # zorder = -1.0
        ax.margins(x = 0.5)
        ax.set_ylim(-0.9, len(df) - 0.1)

        # scatter size legend. we use the *func* argument to supply the inverse of the function
        # used to calculate the sizes from above. The *fmt* ensures to string you want
        handles, labels = sc.legend_elements(
            prop = "sizes", num = 3, color = "gray",
            func = lambda s: 100 * np.sqrt(s) / plt.rcParams["lines.markersize"] / self.scale,
        )

        # ax.legend(
        #     handles, labels, title = "Genes in set",
        #     bbox_to_anchor = (1.02, 0.9),
        #     loc = "upper left",
        #     frameon = False,
        #     labelspacing = 2,
        # )

        ax.set_title(self.title, fontsize = 11, fontweight = "bold")
        self.add_colorbar(sc)

        return ax


    def add_colorbar(self, sc):
        
        cbar = self.fig.colorbar(
            sc, shrink = 0.3, aspect = 5,
            anchor = (0.2, 0), location = "right"
        )
        
        cbar.ax.yaxis.set_tick_params(
            color = "white", direction = "in", 
            left = True, right = True
        )

        # this set slightly larger
        cbar.ax.set_title(self.cbar_title, loc = "left", fontsize = 10, pad = 15)
        for key, spine in cbar.ax.spines.items():
            spine.set_visible(False)


def gsea_dotplot(
    df: pd.DataFrame,
    column: str = "fdr",
    x: Optional[str] = 'nes',
    y: str = "name",
    x_order: Union[List[str], bool] = False,
    y_order: Union[List[str], bool] = False,
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 10,
    terms = None,
    size: float = 5,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 6),
    cmap: str = "turbo",
    ofname: Optional[str] = None,
    xticklabels_rot: Optional[float] = None,
    yticklabels_rot: Optional[float] = None,
    marker: str = "o",
    show_ring: bool = False,
    formatter = lambda x: x,
    **kwargs,
):
    
    dot = dotplot(
        df = df,
        x = x,
        y = y,
        x_order = x_order,
        y_order = y_order,
        hue = column,
        title = title,
        thresh = cutoff,
        n_terms = int(top_term),
        terms = terms,
        dot_scale = size,
        ax = ax,
        figsize = figsize,
        cmap = cmap,
        marker = marker,
        formatter = formatter
    )

    ax = dot.scatter(outer_ring = show_ring)

    if xticklabels_rot:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(xticklabels_rot)

    if yticklabels_rot:
        for label in ax.get_yticklabels():
            label.set_ha("right")
            label.set_rotation(yticklabels_rot)

    return ax.figure


def opa_dotplot(
    df: pd.DataFrame,
    column: str = "fdr",
    x: Optional[str] = 'or',
    y: str = "term",
    x_order: Union[List[str], bool] = False,
    y_order: Union[List[str], bool] = False,
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 10,
    terms = None,
    size: float = 5,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 6),
    cmap: str = "turbo",
    ofname: Optional[str] = None,
    xticklabels_rot: Optional[float] = None,
    yticklabels_rot: Optional[float] = None,
    marker: str = "o",
    show_ring: bool = False,
    formatter = lambda x: x,
    **kwargs,
):
    
    dot = dotplot(
        df = df,
        x = x,
        y = y,
        x_order = x_order,
        y_order = y_order,
        hue = column,
        title = title,
        thresh = cutoff,
        n_terms = int(top_term),
        terms = terms,
        dot_scale = size,
        ax = ax,
        figsize = figsize,
        cmap = cmap,
        marker = marker,
        formatter = formatter
    )

    ax = dot.scatter(outer_ring = show_ring)

    if xticklabels_rot:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(xticklabels_rot)

    if yticklabels_rot:
        for label in ax.get_yticklabels():
            label.set_ha("right")
            label.set_rotation(yticklabels_rot)

    return ax.figure