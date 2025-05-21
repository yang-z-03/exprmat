
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
            top = False, right = False, left = False, labelsize = 9
        )
        ax1.locator_params(axis = "y", nbins = 5)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc, tick_num: "{:.1f}".format(tick_loc))
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
        ax4.text(0.02, 0.05, self._fdr_label, transform = ax4.transAxes, fontsize = 9)
        ax4.text(0.02, 0.15, self._pval_label, transform = ax4.transAxes, fontsize = 9)
        ax4.text(0.02, 0.25, self._nes_label, transform = ax4.transAxes, fontsize = 9)

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