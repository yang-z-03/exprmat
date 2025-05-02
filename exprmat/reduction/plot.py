
import numpy as np
import matplotlib.pyplot as plt
from exprmat.ansi import warning


def embedding(
    adata, basis, color,
    title = None, figsize = (4, 4), ax = None,
    cmap = 'Turbo', legend_loc = 'right margin',frameon = 'small', fontsize = 12
):
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    cvs = ds.Canvas(plot_width = 800, plot_height = 800)
    embedding = adata.obsm[basis]
    df = pd.DataFrame(embedding, columns=['x', 'y'])

    if color in adata.obs.columns:
        labels = adata.obs[color].tolist()
    elif color in adata.var_names:
        X = adata[:, color].X
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)
    elif (not adata.raw is None) and (color in adata.raw.var_names):
        X = adata.raw[:,color].X
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)

    df['label'] = labels
    
    if type(labels[0]) is str:
        df['label'] = df['label'].astype('category')
        agg = cvs.points(df, 'x', 'y', ds.count_cat('label'))
        legend_tag = True

        default_palette = palettes.linear_palette(palettes.all_palettes[cmap][
            list(palettes.all_palettes[cmap].keys())[-1]
        ], len(adata.obs[color].cat.categories))

        if f'{color}.colors' not in adata.uns.keys():
            adata.uns[f'{color}.colors'] = default_palette
        elif len(adata.uns[f'{color}.colors']) != len(default_palette):
            adata.uns[f'{color}.colors'] = default_palette
        
        color_key = dict(zip(
            adata.obs[color].cat.categories,
            adata.uns[f'{color}.colors']
        ))
        
        img = tf.shade(
            tf.spread(agg, px = 0),
            color_key = [color_key[i] for i in df['label'].cat.categories], 
            how = 'eq_hist'
        )
        
    elif (type(labels[0]) is int) or \
         (type(labels[0]) is float) or \
         (type(labels[0]) is np.float32) or \
         (type(labels[0]) is np.float64) or \
         (type(labels[0]) is np.int):
        
        agg = cvs.points(df, 'x', 'y', ds.mean('label'))
        legend_tag = False
        if cmap in palettes.all_palettes.keys():
            num = list(palettes.all_palettes[cmap].keys())[-1]
            img = tf.shade(agg,cmap = palettes.all_palettes[cmap][num])
        else: img = tf.shade(agg,cmap = cmap)
    
    else:
        warning('color label must be categorical (string) or numerical.')
        return None
    
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    else: fig = ax.figure
    
    ax.imshow(img.to_pil(), aspect = 'auto')
    
    def format_coord(x, y): return f"x = {x:.2f}, y = {y:.2f}"
    ax.format_coord = format_coord

    if legend_tag==True:
        unique_labels = adata.obs[color].cat.categories
        for label in unique_labels:
            ax.scatter([], [], c = color_key[label], label = label)
        
        if legend_loc == 'right margin':
            ax.legend(
                frameon = False,
                loc = 'center left',
                bbox_to_anchor = (1, 0.5),
                ncol = (1 if len(unique_labels) <= 14 else 2 if len(unique_labels) <= 30 else 3),
                fontsize = fontsize,
            )

    if frameon == False:
        ax.axis('off')

    elif frameon == 'small':
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_bounds(0,150)
        ax.spines['left'].set_bounds(650,800)
        ax.set_xlabel(f'{basis}1',loc='left',fontsize=fontsize)
        ax.set_ylabel(f'{basis}2',loc='bottom',fontsize=fontsize)

    else:
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_xlabel(f'{basis}1',loc='center',fontsize=fontsize)
        ax.set_ylabel(f'{basis}2',loc='center',fontsize=fontsize)
    
    line_width = 1.2
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    if title is None: title = color
    ax.set_title(title,fontsize = fontsize)

    return ax