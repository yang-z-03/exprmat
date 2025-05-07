
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap as listedcm

from exprmat.ansi import warning, error
from exprmat.utils import setup_styles, plotting_styles


class index_object:
    pass


class index_object_handler:

    def __init__(self):
        self.index = 0
        self.text = ''
        self.color = 'black'

    def set_option(self, index, text, color):
        self.index = index
        self.text = text
        self.color = color
    
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
        patch = mpatches.Ellipse(
            xy = center, 
            width = (height + y0) * 1.5,
            height = (height + y0) * 1.5,
            color = self.color
        )

        annot = mtext.Text(
            x = center[0], y = center[1] - (height + y0) * 0.1, text = str(self.index), color = 'black',
            va = 'center', ha = 'center', # fontproperties = 'bold',
            transform = handlebox.get_transform(), size = 8
        )
        
        annot.set_path_effects([
            mpe.Stroke(linewidth = 2, foreground = 'white'),
            mpe.Normal()
        ])
        
        handlebox.add_artist(patch)
        handlebox.add_artist(annot)
        return patch
    

def embedding_atlas(
    adata, basis, color,
    title = None, figsize = (4, 4), ax = None,
    cmap = 'Turbo', legend_loc = 'right margin',frameon = 'small', fontsize = 10
):
    setup_styles(**plotting_styles)
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

    if legend_tag == True:
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
        # ax.spines['bottom'].set_bounds(0,150)
        # ax.spines['left'].set_bounds(0,150)
        ax.set_xlabel(f'{basis}.1',loc='left',fontsize = fontsize)
        ax.set_ylabel(f'{basis}.2',loc='bottom',fontsize = fontsize)

    else:
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_xlabel(f'{basis}.1',loc='center',fontsize=fontsize)
        ax.set_ylabel(f'{basis}.2',loc='center',fontsize=fontsize)
    
    line_width = 1
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    if title is None: title = color
    ax.set_title(title,fontsize = fontsize)

    return ax


def embedding(
    adata, basis, color,
    
    # query plotting options
    ptsize = 8,
    hue_order = None,
    default_color = 'black',
    alpha = 0.8,
    # palette = 'Turbo',
    rasterize = True,
    sort = True,

    annotate = True,
    annotate_style = 'index',
    annotate_foreground = 'black',
    annotate_stroke = 'white',
    legend = True,

    # contour plotting option.
    contour_plot = True,
    contour_fill = False,
    # contour_hue = None,
    # contour_hue_order = None,
    contour_linewidth = 0.8,
    contour_default_color = 'black',
    # contour_palette = 'hls',
    contour_alpha = 1,
    contour_levels = 10,
    contour_bw = 0.5,

    legend_col = 1,
    add_outline = False,
    outline_color = 'black',

    title = None, figsize = (4, 4), ax = None, dpi = 100, sample_name = None,
    cmap = 'Turbo', cmap_reverse = False, cmap_lower = '#000000',
    legend_loc = 'right margin', frameon = 'small', fontsize = 9
):
    setup_styles(**plotting_styles)
    import pandas as pd
    import seaborn as sb
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    embedding = adata.obsm[basis]
    df = pd.DataFrame(embedding, columns=['x', 'y'])

    if color in adata.obs.columns:
        labels = adata.obs[color].tolist()
    elif color in adata.var_names:
        X = adata[:, color].X
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)
    
    # try some conventions
    elif 'gene' in adata.var.keys() and color in adata.var['gene'].tolist():
        genes = adata.var['gene'].tolist()
        X = adata.X[:, genes.index(color)]
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)

     # try some conventions
    elif 'ensembl' in adata.var.keys() and color in adata.var['ensembl'].tolist():
        genes = adata.var['ensembl'].tolist()
        X = adata.X[:, genes.index(color)]
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)
    
    else: error(f'unable to find feature `{color}` in metadata or variables.')

    df['label'] = labels

    if ax is None:
        fig, axes = plt.subplots(figsize = figsize, dpi = dpi)
    else: axes = ax
    
    atlas_data = {
        'x': df['x'].tolist(), 
        'y': df['y'].tolist(), 
        'edgecolor': None,
        'legend': False,
        'ax': axes,
        'rasterized': True
    }

    if add_outline:
        sb.scatterplot(** atlas_data, color = outline_color, s = ptsize + 40)
        sb.scatterplot(** atlas_data, color = 'white', s = ptsize + 20)

    if type(labels[0]) is str:

        df['label'] = df['label'].astype('category')
        hue = labels
        original_cat = df['label'].value_counts().index.tolist()
        original_cat = sorted(original_cat, key = lambda s: int(s) if str.isdigit(s) else s)
        hue_order = original_cat if hue_order is None else hue_order
        
        default_palette = list(palettes.linear_palette(palettes.all_palettes[cmap][
            list(palettes.all_palettes[cmap].keys())[-1]
        ], len(adata.obs[color].cat.categories))) if isinstance(cmap, str) else cmap

        if f'{color}.colors' not in adata.uns.keys():
            adata.uns[f'{color}.colors'] = default_palette
        elif len(adata.uns[f'{color}.colors']) != len(default_palette):
            adata.uns[f'{color}.colors'] = default_palette
        elif not isinstance(adata.uns[f'{color}.colors'], list):
            adata.uns[f'{color}.colors'] = list(adata.uns[f'{color}.colors'])
        
        default_palette = adata.uns[f'{color}.colors']
        color_key = dict(zip(
            hue_order,
            adata.uns[f'{color}.colors']
        ))
        
        atlas_data['rasterized'] = rasterize
        sb.scatterplot(
            **atlas_data, s = ptsize,
            alpha = alpha, palette = default_palette, color = default_color,
            hue = hue, hue_order = hue_order
        )

        if contour_plot:
            sb.kdeplot(
                x = atlas_data['x'], y = atlas_data['y'], warn_singular = False,
                linewidths = contour_linewidth, bw_adjust = contour_bw, bw_method = 'scott',
                fill = contour_fill, ax = axes, 
                palette = None, color = contour_default_color, alpha = contour_alpha,
                levels = contour_levels, legend = False
            )

        if legend:
            assert len(adata.uns[f'{color}.colors']) == len(hue_order)
            dummy_objects = []
            legend_artists = {}
            for legend_t, legend_c, legend_id in zip(
                hue_order, adata.uns[f'{color}.colors'], range(len(hue_order))
            ):
                dummy = index_object()
                dummy_objects += [dummy]
                handler = index_object_handler()
                handler.set_option(legend_id + 1, legend_t, legend_c)
                legend_artists[dummy] = handler
                pass

            plt.legend(
                dummy_objects, hue_order, handler_map = legend_artists, ncol = legend_col,
                loc = 'upper left', bbox_to_anchor = (1, 1), frameon = False
            )
        

        if annotate:

            for legend_t, legend_c, legend_id in zip(
                hue_order, adata.uns[f'{color}.colors'], range(len(hue_order))
            ):
                # calculate gravity for legend_t class.
                selection = [x == legend_t for x in hue]
                xs = np.array(atlas_data['x'])[selection]
                ys = np.array(atlas_data['y'])[selection]
                center = np.mean(xs), np.mean(ys)
                text = mtext.Text(
                    x = center[0], y = center[1], # fontproperties = 'bold',
                    text = str(legend_id + 1) if annotate_style == 'index' else legend_t, 
                    color = annotate_foreground,
                    ha = 'center', va = 'center', size = 12
                )
                
                text.set_path_effects([
                    mpe.Stroke(linewidth = 3, foreground = annotate_stroke),
                    mpe.Normal()
                ])

                axes.add_artist(text)
                pass
        
    elif (type(labels[0]) is int) or \
         (type(labels[0]) is float) or \
         (type(labels[0]) is np.float32) or \
         (type(labels[0]) is np.float64) or \
         (type(labels[0]) is np.int):
        
        cmap = list(palettes.interp_palette(palettes.all_palettes[cmap][
            list(palettes.all_palettes[cmap].keys())[-1]
        ], 256)) if isinstance(cmap, str) else cmap

        if cmap_reverse:
            cmap = cmap[::-1]

        # np.array([0.95, 0.95, 0.98, 1]) # a tint of bluish gray
        if cmap_lower is not None: 
            for ix in range(5): cmap[ix] = cmap_lower

        cmap = listedcm(cmap)
        
        palette = cmap
        atlas_data['rasterized'] = rasterize
        
        if sort:
            labels = np.array(labels)
            indices = np.argsort(labels)
            labels = labels[indices]
            atlas_data['x'] = np.array(atlas_data['x'])[indices]
            atlas_data['y'] = np.array(atlas_data['y'])[indices]

        sb.scatterplot(
            **atlas_data, s = ptsize,
            alpha = alpha, palette = palette, color = default_color,
            hue = labels, hue_order = None
        )
    
    else:
        warning('color label must be categorical (string) or numerical.')
        return None
    
    def format_coord(x, y): return f"x = {x:.2f}, y = {y:.2f}"
    axes.format_coord = format_coord

    if frameon == False:
        axes.axis('off')

    elif frameon == 'small':
        axes.axis('on')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_visible(True)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        # axes.spines['bottom'].set_bounds(0,1)
        # axes.spines['left'].set_bounds(0,1)
        axes.set_xlabel(f'{basis}.1',loc='left')
        axes.set_ylabel(
            f'{basis}.2' if sample_name is None else
            f'{basis}.2 ({sample_name})',loc = 'bottom'
        )

    else:
        axes.axis('on')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_visible(True)
        axes.spines['top'].set_visible(True)
        axes.spines['right'].set_visible(True)
        axes.set_xlabel(f'{basis}.1',loc='center')
        axes.set_ylabel(
            f'{basis}.2' if sample_name is None else
            f'{basis}.2 ({sample_name})',loc = 'center'
        )
    
    line_width = 0.6
    axes.spines['left'].set_linewidth(line_width)
    axes.spines['bottom'].set_linewidth(line_width)
    if title is None: title = color
    axes.set_title(title)

    if ax is None: return fig
    else: return ax


def gene_gene(
    adata, gene_x, gene_y, color,
    
    # query plotting options
    ptsize = 8,
    hue_order = None,
    default_color = 'black',
    alpha = 0.8,
    # palette = 'Turbo',
    rasterize = True,
    sort = True,

    annotate = True,
    annotate_style = 'index',
    annotate_foreground = 'black',
    annotate_stroke = 'white',
    legend = True,

    # contour plotting option.
    contour_plot = True,
    contour_fill = False,
    # contour_hue = None,
    # contour_hue_order = None,
    contour_linewidth = 0.8,
    contour_default_color = 'black',
    # contour_palette = 'hls',
    contour_alpha = 1,
    contour_levels = 10,
    contour_bw = 0.5,

    legend_col = 1,
    add_outline = False,
    outline_color = 'black',

    title = None, figsize = (4, 4), ax = None, dpi = 100, sample_name = None,
    cmap = 'Turbo', cmap_reverse = False, cmap_lower = '#000000',
    legend_loc = 'right margin', frameon = 'small', fontsize = 9
):
    setup_styles(**plotting_styles)
    import pandas as pd
    import seaborn as sb
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    gx = None
    gy = None
    
    # assign gene X.

    def find_var(gene_name):

        if gene_name in adata.obs.columns:
            vec = adata.obs[gene_name].tolist()

        elif gene_name in adata.var_names:
            X = adata[:, gene_name].X
            if issparse(X): vec = X.toarray().reshape(-1)
            else: vec = X.reshape(-1)

        # try some conventions
        elif 'gene' in adata.var.keys() and gene_name in adata.var['gene'].tolist():
            genes = adata.var['gene'].tolist()
            X = adata.X[:, genes.index(gene_name)]
            if issparse(X): vec = X.toarray().reshape(-1)
            else: vec = X.reshape(-1)

         # try some conventions
        elif 'ensembl' in adata.var.keys() and gene_name in adata.var['ensembl'].tolist():
            genes = adata.var['ensembl'].tolist()
            X = adata.X[:, genes.index(gene_name)]
            if issparse(X): vec = X.toarray().reshape(-1)
            else: vec = X.reshape(-1)

        else: error(f'unable to find gene `{gene_name}` in metadata or variables.')
        return vec
    
    gx = find_var(gene_x)
    gy = find_var(gene_y)
    labels = find_var(color)

    df = pd.DataFrame({
        'x': gx,
        'y': gy,
        'label': labels
    })

    if ax is None:
        fig, axes = plt.subplots(figsize = figsize, dpi = dpi)
    else: axes = ax
    
    atlas_data = {
        'x': df['x'].tolist(), 
        'y': df['y'].tolist(), 
        'edgecolor': None,
        'legend': False,
        'ax': axes,
        'rasterized': True
    }

    if add_outline:
        sb.scatterplot(** atlas_data, color = outline_color, s = ptsize + 40)
        sb.scatterplot(** atlas_data, color = 'white', s = ptsize + 20)

    if type(labels[0]) is str:

        df['label'] = df['label'].astype('category')
        hue = labels
        original_cat = df['label'].value_counts().index.tolist()
        original_cat = sorted(original_cat, key = lambda s: int(s) if str.isdigit(s) else s)
        hue_order = original_cat if hue_order is None else hue_order
        
        default_palette = list(palettes.linear_palette(palettes.all_palettes[cmap][
            list(palettes.all_palettes[cmap].keys())[-1]
        ], len(adata.obs[color].cat.categories))) if isinstance(cmap, str) else cmap

        if f'{color}.colors' not in adata.uns.keys():
            adata.uns[f'{color}.colors'] = default_palette
        elif len(adata.uns[f'{color}.colors']) != len(default_palette):
            adata.uns[f'{color}.colors'] = default_palette
        elif not isinstance(adata.uns[f'{color}.colors'], list):
            adata.uns[f'{color}.colors'] = list(adata.uns[f'{color}.colors'])

        color_key = dict(zip(
            hue_order,
            adata.uns[f'{color}.colors']
        ))
        
        atlas_data['rasterized'] = rasterize
        sb.scatterplot(
            **atlas_data, s = ptsize,
            alpha = alpha, palette = default_palette, color = default_color,
            hue = hue, hue_order = hue_order
        )

        if contour_plot:
            sb.kdeplot(
                x = atlas_data['x'], y = atlas_data['y'], warn_singular = False,
                linewidths = contour_linewidth, bw_adjust = contour_bw, bw_method = 'scott',
                fill = contour_fill, ax = axes, 
                palette = None, color = contour_default_color, alpha = contour_alpha,
                levels = contour_levels, legend = False
            )

        if legend:
            assert len(adata.uns[f'{color}.colors']) == len(hue_order)
            dummy_objects = []
            legend_artists = {}
            for legend_t, legend_c, legend_id in zip(
                hue_order, adata.uns[f'{color}.colors'], range(len(hue_order))
            ):
                dummy = index_object()
                dummy_objects += [dummy]
                handler = index_object_handler()
                handler.set_option(legend_id + 1, legend_t, legend_c)
                legend_artists[dummy] = handler
                pass

            plt.legend(
                dummy_objects, hue_order, handler_map = legend_artists, ncol = legend_col,
                loc = 'upper left', bbox_to_anchor = (1, 1), frameon = False
            )
        

        if annotate:

            for legend_t, legend_c, legend_id in zip(
                hue_order, adata.uns[f'{color}.colors'], range(len(hue_order))
            ):
                # calculate gravity for legend_t class.
                selection = [x == legend_t for x in hue]
                xs = np.array(atlas_data['x'])[selection]
                ys = np.array(atlas_data['y'])[selection]
                center = np.mean(xs), np.mean(ys)
                text = mtext.Text(
                    x = center[0], y = center[1], # fontproperties = 'bold',
                    text = str(legend_id + 1) if annotate_style == 'index' else legend_t, 
                    color = annotate_foreground,
                    ha = 'center', va = 'center', size = 12
                )
                
                text.set_path_effects([
                    mpe.Stroke(linewidth = 3, foreground = annotate_stroke),
                    mpe.Normal()
                ])

                axes.add_artist(text)
                pass
        
    elif (type(labels[0]) is int) or \
         (type(labels[0]) is float) or \
         (type(labels[0]) is np.float32) or \
         (type(labels[0]) is np.float64) or \
         (type(labels[0]) is np.int):
        
        cmap = list(palettes.interp_palette(palettes.all_palettes[cmap][
            list(palettes.all_palettes[cmap].keys())[-1]
        ], 256)) if isinstance(cmap, str) else cmap

        if cmap_reverse:
            cmap = cmap[::-1]

        # np.array([0.95, 0.95, 0.98, 1]) # a tint of bluish gray
        if cmap_lower is not None: 
            for ix in range(5): cmap[ix] = cmap_lower

        cmap = listedcm(cmap)
        
        palette = cmap
        atlas_data['rasterized'] = rasterize
        
        if sort:
            labels = np.array(labels)
            indices = np.argsort(labels)
            labels = labels[indices]
            atlas_data['x'] = np.array(atlas_data['x'])[indices]
            atlas_data['y'] = np.array(atlas_data['y'])[indices]

        sb.scatterplot(
            **atlas_data, s = ptsize,
            alpha = alpha, palette = palette, color = default_color,
            hue = labels, hue_order = None
        )
    
    else:
        warning('color label must be categorical (string) or numerical.')
        return None
    
    def format_coord(x, y): return f"x = {x:.2f}, y = {y:.2f}"
    axes.format_coord = format_coord

    if frameon == False:
        axes.axis('off')

    elif frameon == 'small':
        axes.axis('on')
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_visible(True)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        # axes.spines['bottom'].set_bounds(0,1)
        # axes.spines['left'].set_bounds(0,1)
        axes.set_xlabel(gene_x,loc='left')
        axes.set_ylabel(
            gene_y if sample_name is None else
            f'{gene_y} ({sample_name})',loc = 'bottom'
        )

    else:
        axes.axis('on')
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_visible(True)
        axes.spines['top'].set_visible(True)
        axes.spines['right'].set_visible(True)
        axes.set_xlabel(gene_x,loc='center')
        axes.set_ylabel(
            gene_y if sample_name is None else
            f'{gene_y} ({sample_name})',loc = 'center'
        )
    
    line_width = 0.6
    axes.spines['left'].set_linewidth(line_width)
    axes.spines['bottom'].set_linewidth(line_width)
    if title is None: title = color
    axes.set_title(title)

    if ax is None: return fig
    else: return ax