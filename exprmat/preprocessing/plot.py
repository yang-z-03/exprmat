
from exprmat.utils import setup_styles, plotting_styles


def rna_plot_qc_metrics(adata, sample_name, figsize = (10, 2.2)):

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    setup_styles(**plotting_styles)
    fig, axes = plt.subplots(1, 4)

    axes[0].hist(adata.obs['n.umi'].tolist(), bins = 100)
    axes[0].set_xlabel('Reads')
    axes[0].set_ylabel(f'Sample {sample_name}' + '\nFrequency')

    axes[1].hist(adata.obs['n.genes'].tolist(), bins = 100)
    axes[1].set_xlabel('Gene detection')
    axes[1].set_ylabel('Frequency')

    doublets = adata.obs[adata.obs['is.doublet']]
    colors = {
        True: 'black',
        False: 'gray'
    }

    axes[2].scatter(
        adata.obs['n.genes'].tolist(),
        adata.obs['pct.mito'].tolist(),
        c = adata.obs['qc'].map(colors), s = 2
    )

    axes[2].scatter(
        doublets['n.genes'].tolist(),
        doublets['pct.mito'].tolist(),
        c = 'red', s = 2
    )
    axes[2].set_xlabel('Gene detection')
    axes[2].set_ylabel('Mitochondrial transcripts')

    axes[3].scatter(
        adata.obs['n.genes'].tolist(),
        adata.obs['n.umi'].tolist(),
        c = adata.obs['qc'].map(colors), s = 2
    )
    
    axes[3].scatter(
        doublets['n.genes'].tolist(),
        doublets['n.umi'].tolist(),
        c = 'red', s = 2
    )
    axes[3].set_xlabel('Gene detection')
    axes[3].set_ylabel('Reads')

    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.set_dpi(100)
    fig.tight_layout(pad = 0.1)

    return fig