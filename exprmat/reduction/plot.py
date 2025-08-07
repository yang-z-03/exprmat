
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap as listedcm

from exprmat.ansi import warning, error
from exprmat.utils import setup_styles, choose_layer


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
    hue_order = None,
    title = None, figsize = (4, 4), ax = None,
    cmap = 'turbo', cmap_reverse = False, cmap_lower = '#000000',
    legend = True, legend_loc = 'right margin', legend_col = 1, 
    frameon = 'small', fontsize = 10,
    annotate = True, annotate_style = 'index', annot_size = 12,
    annotate_foreground = 'black', annotate_stroke = 'white',
    dpi = 100, sample_name = None
):
    res_factor = 1.5
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    cvs = ds.Canvas(plot_width = int(figsize[0] * dpi * res_factor), plot_height = int(figsize[1] * dpi * res_factor))
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
    
    if ax is None: fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    else: fig = ax.figure

    if type(labels[0]) is str:
        df['label'] = df['label'].astype('category')
        agg = cvs.points(df, 'x', 'y', ds.count_cat('label'))
        legend_tag = legend

        original_cat = df['label'].value_counts().index.tolist()
        original_cat = sorted(original_cat, key = lambda s: s.zfill(8) if str.isdigit(s) else s)
        hue_order = original_cat if hue_order is None else hue_order
        
        default_palette = palettes.get_palette(cmap, len(adata.obs[color].cat.categories)) \
            if isinstance(cmap, str) else cmap

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
        
        img = tf.shade(
            tf.spread(agg, px = 0),
            color_key = [color_key[i] for i in df['label'].cat.categories], 
            how = 'eq_hist'
        )

        ax.imshow(
            img.to_pil(), 
            aspect = 'auto', 
            interpolation = 'bicubic',
            extent = (0, dpi * figsize[0] * res_factor, 0, dpi * figsize[1] * res_factor)
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
            
            scale_max_x = dpi * figsize[0] * res_factor
            scale_max_y = dpi * figsize[1] * res_factor
            min_x, min_y, max_x, max_y = np.min(df['x']), np.min(df['y']), np.max(df['x']), np.max(df['y'])

            for legend_t, legend_c, legend_id in zip(
                hue_order, adata.uns[f'{color}.colors'], range(len(hue_order))
            ):
                # calculate gravity for legend_t class.
                xs = np.array(df.loc[df['label'] == legend_t, 'x'])
                ys = np.array(df.loc[df['label'] == legend_t, 'y'])
                
                mx, my = np.mean(xs), np.mean(ys)
                mx = scale_max_x * ((mx - min_x) / (max_x - min_x))
                my = scale_max_y * ((my - min_y) / (max_y - min_y))
                text = mtext.Text(
                    x = mx, y = my, # fontproperties = 'bold',
                    text = str(legend_id + 1) if annotate_style == 'index' else legend_t, 
                    color = annotate_foreground,
                    ha = 'center', va = 'center', size = annot_size
                )
                
                text.set_path_effects([
                    mpe.Stroke(linewidth = 3, foreground = annotate_stroke),
                    mpe.Normal()
                ])

                ax.add_artist(text)
                pass
        

    elif (type(labels[0]) is int) or \
         (type(labels[0]) is float) or \
         (type(labels[0]) is np.float32) or \
         (type(labels[0]) is np.float64) or \
         (type(labels[0]) is np.int):
        
        agg = cvs.points(df, 'x', 'y', ds.mean('label'))
        legend_tag = False

        cmap = palettes.get_palette(cmap, 256) if isinstance(cmap, str) else cmap

        if cmap_reverse:
            cmap = cmap[::-1]

        # np.array([0.95, 0.95, 0.98, 1]) # a tint of bluish gray
        if cmap_lower is not None: 
            for ix in range(5): cmap[ix] = cmap_lower

        cmap = listedcm(cmap)
        img = tf.shade(agg,cmap = cmap)
        
        ax.imshow(
            img.to_pil(), 
            aspect = 'auto', 
            interpolation = 'bicubic',
            extent = (0, dpi * figsize[0] * res_factor, 0, dpi * figsize[1] * res_factor)
        )
    
    else:
        warning('color label must be categorical (string) or numerical.')
        return None
    

    def format_coord(x, y): return f"x = {x:.2f}, y = {y:.2f}"
    ax.format_coord = format_coord

    # if legend_tag == True:
    #     unique_labels = adata.obs[color].cat.categories
    #     for label in unique_labels:
    #         ax.scatter([], [], c = color_key[label], label = label)
    #     
    #     if legend_loc == 'right margin':
    #         ax.legend(
    #             frameon = False,
    #             loc = 'center left',
    #             bbox_to_anchor = (1, 0.5),
    #             ncol = (1 if len(unique_labels) <= 14 else 2 if len(unique_labels) <= 30 else 3),
    #             fontsize = fontsize,
    #         )

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
    
    line_width = 0.6
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    if title is None: title = color
    ax.set_title(title,fontsize = fontsize * 1.2)
    ax.set_xlim(dpi * figsize[0] * -0.05 * res_factor, dpi * figsize[0] * 1.05 * res_factor)
    ax.set_ylim(dpi * figsize[1] * -0.05 * res_factor, dpi * figsize[1] * 1.05 * res_factor)

    return ax.figure


def embedding(
    adata, basis, color,
    slot = 'X',
    
    # query plotting options
    ptsize = 8,
    size = None,
    hue_order = None,
    default_color = 'black',
    alpha = 0.8,
    # palette = 'turbo',
    rasterize = True,
    sort = True,

    annotate = True,
    annotate_style = 'index',
    annotate_foreground = 'black',
    annotate_stroke = 'white',
    annotate_fontsize = 12,
    annotate_only = None,
    legend = True,

    # contour plotting option.
    contour_plot = True,
    contour_fill = False,
    contour_mask = None,
    contour_mask_values = [],
    contour_linewidth = 0.8,
    contour_default_color = 'black',
    # contour_palette = 'hls',
    contour_alpha = 1,
    contour_levels = 10,
    contour_bw = 0.5,

    legend_col = 1,
    add_outline = False,
    outline_color = 'black',
    outline_margin = 20,

    title = None, figsize = (4, 4), ax = None, dpi = 100, sample_name = None,
    cmap = 'turbo', cmap_reverse = False, cmap_lower = '#000000',
    hue_norm = None, 
    ticks = False,
    frameon = 'small',
    xlabel = None, ylabel = None
):
    
    setup_styles()
    import pandas as pd
    import seaborn as sb
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    if isinstance(basis, str):
        embedding = adata.obsm[basis]
    elif isinstance(basis, pd.DataFrame):
        embedding = basis.data 
    elif isinstance(basis, np.matrix) or isinstance(basis, np.ndarray):
        embedding = basis

    df = pd.DataFrame(embedding, columns = ['x', 'y'])

    if color in adata.obs.columns:
        labels = adata.obs[color].tolist()
    elif color in adata.var_names:
        X = choose_layer(adata[:, color], layer = slot)
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)
    
    # try some conventions
    elif 'gene' in adata.var.keys() and color in adata.var['gene'].tolist():
        genes = adata.var['gene'].tolist()
        X = choose_layer(adata[:, genes.index(color)], layer = slot)
        if issparse(X): labels = X.toarray().reshape(-1)
        else: labels = X.reshape(-1)

     # try some conventions
    elif 'ensembl' in adata.var.keys() and color in adata.var['ensembl'].tolist():
        genes = adata.var['ensembl'].tolist()
        X = choose_layer(adata[:, genes.index(color)], layer = slot)
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
        sb.scatterplot(** atlas_data, color = outline_color, s = ptsize + outline_margin + 20)
        sb.scatterplot(** atlas_data, color = 'white', s = ptsize + outline_margin)

    if type(labels[0]) is str:

        # eliminate nan's in the labels for categories
        df['label'] = df['label'].fillna('n/a').astype('str').astype('category')
        labels = df['label'].tolist()

        hue = labels
        original_cat = df['label'].value_counts().index.tolist()
        original_cat = sorted(original_cat, key = lambda s: s.zfill(8) if str.isdigit(s) else s)
        hue_order = original_cat if hue_order is None else hue_order
        
        if cmap is not None:
            
            default_palette = palettes.get_palette(cmap, len(adata.obs[color].cat.categories)) \
                if isinstance(cmap, str) else cmap

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
        
        else: default_palette = None
        
        atlas_data['rasterized'] = rasterize
        sb.scatterplot(
            **atlas_data, s = ptsize if size is None else (ptsize * adata.obs[size]).tolist(),
            alpha = alpha, palette = default_palette, color = default_color,
            hue = hue if default_palette is not None else None, 
            hue_order = hue_order if default_palette is not None else None
        )

        if contour_plot:

            cx = atlas_data['x']
            cy = atlas_data['y']
            if contour_mask is not None:
                cx = [y for x, y in zip(adata.obs[contour_mask], cx) if x in contour_mask_values]
                cy = [y for x, y in zip(adata.obs[contour_mask], cy) if x in contour_mask_values]

            sb.kdeplot(
                x = cx, y = cy, warn_singular = False,
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
                hue_order, 
                adata.uns[f'{color}.colors'] if cmap is not None else ['black'] * len(hue_order), 
                range(len(hue_order))
            ):
                if annotate_only is not None:
                    if legend_t not in annotate_only:
                        continue
                    
                # calculate gravity for legend_t class.
                selection = [x == legend_t for x in hue]
                xs = np.array(atlas_data['x'])[selection]
                ys = np.array(atlas_data['y'])[selection]
                center = np.mean(xs), np.mean(ys)
                text = mtext.Text(
                    x = center[0], y = center[1], # fontproperties = 'bold',
                    text = str(legend_id + 1) if annotate_style == 'index' else legend_t, 
                    color = annotate_foreground,
                    ha = 'center', va = 'center', size = annotate_fontsize
                )
                
                text.set_path_effects([
                    mpe.Stroke(linewidth = 3, foreground = annotate_stroke),
                    mpe.Normal()
                ])

                axes.add_artist(text)
                pass
    
    elif type(labels[0]) is bool:

        sb.scatterplot(
            data = df.loc[df['label'] == False, :], x = 'x', y = 'y',
            edgecolor = None, legend = False, ax = axes, s = ptsize, rasterized = rasterize,
            alpha = alpha, color = '#f0f0f0'
        )

        sb.scatterplot(
            data = df.loc[df['label'] == True, :], x = 'x', y = 'y',
            edgecolor = None, legend = False, ax = axes, s = ptsize, rasterized = rasterize,
            alpha = alpha, color = 'red'
        )

    elif (type(labels[0]) is int) or \
         (type(labels[0]) is float) or \
         (type(labels[0]) is np.float32) or \
         (type(labels[0]) is np.float64) or \
         (type(labels[0]) is np.uint32):
        
        cmap = palettes.get_palette(cmap, 256) if isinstance(cmap, str) else cmap

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

        # if labels is all identical
        if len(np.unique(labels)) <= 1:
            palette = None
            labels = None

        sb.scatterplot(
            **atlas_data, s = ptsize,
            alpha = alpha, palette = palette, color = default_color,
            hue = labels, hue_order = None, hue_norm = hue_norm
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
        if not ticks: axes.set_xticks([])
        if not ticks: axes.set_yticks([])
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
        if not ticks: axes.set_xticks([])
        if not ticks: axes.set_yticks([])
        axes.spines['left'].set_visible(True)
        axes.spines['bottom'].set_visible(True)
        axes.spines['top'].set_visible(True)
        axes.spines['right'].set_visible(True)
        axes.set_xlabel(f'{basis}.1',loc='center')
        axes.set_ylabel(
            f'{basis}.2' if sample_name is None else
            f'{basis}.2 ({sample_name})',loc = 'center'
        )
    
    if xlabel is not None: axes.set_xlabel(xlabel)
    if ylabel is not None: axes.set_ylabel(ylabel)

    line_width = 0.6
    axes.spines['left'].set_linewidth(line_width)
    axes.spines['bottom'].set_linewidth(line_width)
    if title is None: title = color
    axes.set_title(title)

    if ax is None: return fig
    else: return ax


def gene_gene(
    adata, gene_x, gene_y, color,
    layer = 'X',
    
    # query plotting options
    ptsize = 8,
    hue_order = None,
    default_color = 'black',
    alpha = 0.8,
    # palette = 'turbo',
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
    cmap = 'turbo', cmap_reverse = False, cmap_lower = '#000000',
    legend_loc = 'right margin', frameon = 'small', fontsize = 9,

    remove_zero_expression = False,
    scale = 'asis', arcsinh_divider = None
):
    setup_styles()
    import pandas as pd
    import seaborn as sb
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    gx = None
    gy = None
    
    # assign gene X.
    from exprmat.utils import find_variable as find_var
    gx = find_var(adata, gene_name = gene_x, layer = layer)
    gy = find_var(adata, gene_name = gene_y, layer = layer)
    labels = find_var(adata, color) if color is not None else ['.'] * len(gx)

    df = pd.DataFrame({
        'x': gx,
        'y': gy,
        'label': labels
    })

    if remove_zero_expression:
        df = df.loc[(df['x'] > 0) & (df['y'] > 0), :].copy()

    if scale == 'asis': pass
    elif scale == 'log':
        df['x'] = np.log1p(df['x'])
        df['y'] = np.log1p(df['y'])
    elif scale == 'expm1':
        df['x'] = np.expm1(df['x'])
        df['y'] = np.expm1(df['y'])
    elif scale == 'arcsinh':
        lx = np.expm1(df['x']).loc[df['x'] > 0]
        ly = np.expm1(df['y']).loc[df['x'] > 0]
        df['x'] = np.arcsinh(np.expm1(df['x']) / (
            arcsinh_divider if arcsinh_divider is not None 
            else (np.median(lx) / 20)))
        df['y'] = np.arcsinh(np.expm1(df['y']) / (
            arcsinh_divider if arcsinh_divider is not None 
            else (np.median(ly) / 20)))

    gx = df['x'].tolist()
    gy = df['y'].tolist()
    labels = df['label'].tolist()

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
        hue_order = None
        default_palette = None

        if color is not None:
            original_cat = df['label'].value_counts().index.tolist()
            original_cat = sorted(original_cat, key = lambda s: s.zfill(8) if str.isdigit(s) else s)
            hue_order = original_cat if hue_order is None else hue_order

            default_palette = palettes.get_palette(cmap, len(adata.obs[color].cat.categories)) \
                if isinstance(cmap, str) else cmap

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
            hue = hue if color is not None else None, hue_order = hue_order
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
            assert len(adata.uns[f'{color}.colors']) >= len(hue_order)
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
        
        cmap = palettes.get_palette(cmap, 256) if isinstance(cmap, str) else cmap

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


def gene_gene_regress(
    atlas, x, y, figsize = (3, 3), dpi = 100, 
    title = None, regression_info = True, layer = 'X',
    scale = 'asis'
):
    
    fig, axis = plt.subplots(1, 1, figsize = figsize, dpi = dpi)

    if isinstance(x, str):
        xlab = f'Log expression {x}'; x = [x]
    else: xlab = f'Log expression of signature \n{", ".join(x)}'

    if isinstance(x, str):
        ylab = f'Log expression {y}'; y = [y]
    else: ylab = f'Log expression of signature \n{", ".join(y)}'
    
    from exprmat.utils import find_variable as find_var
    
    def get_gene_signature(atlas, genes, layer = 'X'):
        x_genes = [find_var(atlas, x, layer) for x in genes]
        x_genes = np.stack(x_genes, axis = 1)
        x_genes = np.mean(x_genes, axis = 1)
        return x_genes

    x_genes = get_gene_signature(atlas, x, layer = layer)
    y_genes = get_gene_signature(atlas, y, layer = layer)
    
    if scale == 'asis': pass
    elif scale == 'log':
        x_genes = np.log1p(x_genes)
        y_genes = np.log1p(y_genes)

    sns.regplot(
        x = (x_genes), 
        y = (y_genes), 
        ax = axis,
        scatter_kws = { 's': 4, 'color': 'black' },
        line_kws = { 'linewidth': 1, 'color': 'red' }
    )

    axis.set_xlabel(xlab, loc = 'left', linespacing = 1.5)
    axis.set_ylabel(ylab, loc = 'bottom', linespacing = 1.5)

    for pos in ['right', 'top']:
        axis.spines[pos].set_visible(False)

    from scipy import stats
    res = stats.linregress(x_genes, y_genes)
    
    r = (f"R: {res.rvalue:5.3f}")
    p = (f"p: {res.pvalue:.2e}")
    if title is not None: axis.set_title(title)

    if regression_info:
        x1, x2 = axis.get_xlim()
        y1, y2 = axis.get_ylim()
    
        axis.text(
            s = r + '\n' + p, va = 'bottom', ha = 'right', 
            x = x1 + 0.95 * (x2 - x1), 
            y = y1 + 0.05 * (y2 - y1),
            linespacing = 1.5
        )

    return fig