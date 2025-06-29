
import matplotlib.pyplot as plt
import pandas as pd


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
    adata, k, nmf_slot = 'cnmf', metrics = 'cosine', downsample = 0.1,
    cmap = 'Reds_r', figsize = (3, 3), dpi = 100
):
    
    import numpy as np
    fig = plt.figure(figsize = figsize, dpi = dpi)
    df = adata.obsm[f'{nmf_slot}.{k}']
    nrow = df.shape[0]
    subset = np.random.choice(range(nrow), int(nrow * downsample), replace = False)
    from exprmat.clustering.seriation import pdist
    pdistc = pdist(df.iloc[subset, :], metric = 'cosine')

    from exprmat.clustering.seriation import serial_matrix, squareform
    serial, _, _ = serial_matrix(squareform(pdistc), method = 'single')
    fig.gca().imshow(serial, cmap = cmap)

    return fig
