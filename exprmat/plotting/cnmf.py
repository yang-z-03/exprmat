
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def cnmf_silhoutte(adata, nmf_slot = 'cnmf', figsize = (3, 2), dpi = 100):

    fig = plt.figure(figsize = figsize, dpi = dpi)
    df = adata.uns[f'{nmf_slot}.stats']
    fig.gca().plot(df.ncomps, df.silhoutte, color = 'k')

    # fig.gca().set_ylim([-0.1, 1.1])
    # fig.gca().set_yticks([0, 0.25, 0.5, 0.75, 1])
    fig.gca().set_ylabel('Silhoutte score')

    fig.gca().set_xlabel('NMF components')
    fig.gca().set_xticks(df.ncomps)

    for pos in ['right', 'top']:
        fig.gca().spines[pos].set_visible(False)

    return fig


def cnmf_density(adata, k, nmf_slot = 'cnmf', figsize = (3, 2), dpi = 100):
    
    fig = plt.figure(figsize = figsize, dpi = dpi)
    df = adata.uns[f'{nmf_slot}.density.{k}']
    fig.gca().hist(df, bins = 100, color = 'k')

    fig.gca().set_ylabel('Local density')
    fig.gca().set_xlabel('Distance')

    for pos in ['right', 'top']:
        fig.gca().spines[pos].set_visible(False)

    return fig


def cnmf_distance_comps(adata, k, nmf_slot = 'cnmf', cmap = 'Reds_r', figsize = (3, 3), dpi = 100):
    
    fig = plt.figure(figsize = figsize, dpi = dpi)
    df = adata.uns[f'{nmf_slot}.dist.{k}']
    fig.gca().imshow(df, cmap = cmap)

    return fig


def cnmf_distance_usages(
    adata, k, nmf_slot = 'cnmf', metrics = 'cosine', downsample = 0.1, method = 'single',
    annotations = None, show_indices = False, legend_cols = 1,
    cmap = 'Reds_r', cmap_annotations = 'Set1', figsize = (3, 3), dpi = 100
):
    
    import numpy as np

    df = adata.obsm[f'{nmf_slot}.{k}']
    nrow = df.shape[0]
    subset = np.random.choice(range(nrow), int(nrow * downsample), replace = False)
    from exprmat.clustering.seriation import pdist
    data = df.iloc[subset, :].copy()
    pdistc = pdist(data, metric = metrics)

    from exprmat.clustering.seriation import serial_matrix, squareform
    serial, order, _ = serial_matrix(squareform(pdistc), method = method)
    indices = adata.obs.iloc[subset, :].iloc[order, :].index.tolist()
    if annotations:
        if annotations in adata.obs.columns: 
            annotations = adata.obs.iloc[subset, :].iloc[order, :][annotations].tolist()
        elif annotations in adata.obsm.keys(): 
            annotations = adata[subset, :][order, :].obsm[annotations]

    return matrix_plot(
        serial, labels = indices, annotations = annotations, cmap_annotations = cmap_annotations, 
        cmap = cmap, figsize = figsize, dpi = dpi, show_indices = show_indices,
        legend_cols = legend_cols
    )


def cnmf_distance_modules(
    adata, k, nmf_slot = 'cnmf', metrics = 'cosine', downsample = 1, method = 'complete',
    annotations = None, cmap_annotations = 'Set1', show_indices = False,
    cmap = 'Reds_r', figsize = (3, 3), dpi = 100, legend_cols = 1
):
    
    import numpy as np
    df = adata.varm[f'{nmf_slot}.{k}']
    nrow = df.shape[0]
    subset = np.random.choice(range(nrow), int(nrow * downsample), replace = False)
    from exprmat.clustering.seriation import pdist
    data = df[subset, :].copy()
    pdistc = pdist(data, metric = metrics)

    from exprmat.clustering.seriation import serial_matrix, squareform
    serial, order, _ = serial_matrix(squareform(pdistc), method = method)
    indices = adata.var.iloc[subset, :].iloc[order, :].index.tolist()
    if annotations: 
        if annotations in adata.var.columns: 
            annotations = adata.var.iloc[subset, :].iloc[order, :][annotations].tolist()
        elif annotations in adata.varm.keys(): 
            annotations = adata[:, subset][:, order].varm[annotations]

    return matrix_plot(
        serial, labels = indices, annotations = annotations, cmap_annotations = cmap_annotations, 
        cmap = cmap, figsize = figsize, dpi = dpi, show_indices = show_indices, legend_cols = legend_cols
    )


def matrix_plot(
    mat, labels = None, annotations = None, 
    cmap = 'Reds_r', cmap_annotations = 'Set1',
    show_indices = True,
    figsize = (3, 3), dpi = 100, legend_cols = 1
):
    fig = plt.figure(figsize = figsize, dpi = dpi)
    fig.gca().imshow(mat, cmap = cmap)
    ax = fig.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    if labels is not None and show_indices:
        ax.set_yticks([x for x in range(len(labels))])
        ax.set_yticklabels(labels)
        ax.set_xticks([])

    # categorical annotations
    if annotations is not None and isinstance(annotations, list):
        ax_annot = ax.inset_axes([1.02, 0, 0.05, 1], sharey = ax)
        annot_u = list(set(annotations))
        annot_n = [annot_u.index(x) for x in annotations]

        from exprmat.plotting import palettes
        colors = list(palettes.linear_palette(palettes.all_palettes[cmap_annotations][
            list(palettes.all_palettes[cmap_annotations].keys())[-1]
        ], len(annot_u)))
        colormap = {x: y for x, y in zip(annot_u, colors)}

        ax_annot.set_xlim(-0.5, 0.5)
        ax_annot.set_ylim(len(annot_n) - 0.5, -0.5)
        for idx, x in enumerate(annot_n):
            ax_annot.fill_between([-0.5, 0.5], [idx - 0.5, idx - 0.5], [idx + 0.5, idx + 0.5], color = colormap[annot_u[x]])

        # ax_annot.set_yticks([])
        ax_annot.set_xticks([0])
        ax_annot.set_xticklabels([''])
        ax_annot.tick_params(axis = "y", labelleft = False, length = 0)
        
        from matplotlib.lines import Line2D as line2d

        legend_elements = [
            line2d([0], [0], marker = 's', color = 'w', label = alab,
                markerfacecolor = acol, markersize = 8)
            for alab, acol in zip(annot_u, colors)
        ]

        ax_annot.legend(
            handles = legend_elements, 
            bbox_to_anchor = (1.0, 1), 
            loc = 'upper left', 
            borderaxespad = 0, frameon = False,
            ncol = legend_cols
        )
    
    # numerical annotations (from obsm or varm)
    elif annotations is not None and (
        isinstance(annotations, np.ndarray) or
        isinstance(annotations, np.matrix) or
        isinstance(annotations, pd.DataFrame)
    ):
        xn = None
        if isinstance(annotations, pd.DataFrame):
            xn = annotations.columns.tolist()
            annotations = annotations.values

        ncol = annotations.shape[1]
        ax_annot = ax.inset_axes([1.05, 0, max(0.3, 0.07 * ncol), 1], sharey = ax)
        
        ax_annot.imshow(annotations, cmap = cmap_annotations, aspect = 'auto', interpolation = 'nearest')
        # ax_annot.set_yticks([])
        if xn: 
            ax_annot.set_xticks([x for x in range(len(xn))])
            ax_annot.set_xticklabels(xn, rotation = 90)
        else:
            ax_annot.set_xticks([0])
            ax_annot.set_xticklabels([''])

        ax_annot.tick_params(axis = "y", labelleft = False, length = 0)
    
    return fig