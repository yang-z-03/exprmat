
import numpy as np
import matplotlib.pyplot as plt
import math

from exprmat.plotting.utils import line, density


def fragment_size(adata, ax = None, figsize = (3, 2), dpi = 100):
    return line(
        np.arange(len(adata.uns['frag.sizes'])) + 1, 
        adata.uns['frag.sizes'],
        xlabel = 'Fragment size (bp)',
        ylabel = 'N. fragments',
        figsize = figsize, dpi = dpi,
        ax = ax
    )


def tss_profile(adata, ax = None, figsize = (3, 2), dpi = 100):
    return line(
        np.arange(len(adata.uns['tss.profile'])) - 2000, 
        adata.uns['tss.profile'],
        xlabel = 'Relative offset to TSS center (bp)',
        ylabel = 'N. fragments',
        figsize = figsize, dpi = dpi,
        ax = ax
    )


def tsse(adata, ax = None, figsize = (3, 3), dpi = 100):
    return density(
        adata.obs['n.fragments'], 
        adata.obs['tsse'], 
        xscale = lambda x: np.log10(x),
        xlabel = 'lg(N) fragments',
        ylabel = 'TSS enrichment score',
        size = 2,
        ax = ax
    )


def atac_qc_metrics(adata, sample_name, figsize = (8, 2.5), dpi = 100):

    from exprmat.descriptive.atac import fragment_size_distribution, frip, tss_enrichment
    fragment_size_distribution(adata)
    tss_enrichment(adata, adata.uns['assembly'])

    fig, axes = plt.subplots(1, 3, figsize = figsize, dpi = dpi)
    fragment_size(adata, axes[0])
    axes[0].set_ylabel(f'Sample {sample_name}\nN. fragments')
    tss_profile(adata, axes[1])
    tsse(adata, axes[2])

    return fig