
import sys
import operator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from exprmat.ansi import error


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
        ax.set_yticklabels(ylabels if self.yticklabels else "", fontsize=14)
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