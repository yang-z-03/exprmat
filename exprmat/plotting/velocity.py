
import numpy as np

import matplotlib.pyplot as pl
from matplotlib import rcParams
from scipy.stats import norm as normal
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from collections import abc

from exprmat.dynamics.linreg import sum
from exprmat.dynamics.utils import groups_to_bool
from exprmat.dynamics.velocity import quiver_autoscale, velocity_embedding
from exprmat.dynamics.generank import rank_velocity_genes
from exprmat.dynamics.moments import second_order_moments
from exprmat.plotting import embedding
from exprmat.utils import translate_variables
from exprmat.ansi import error, warning, info


def make_unique_list(key, allow_array = False):

    if isinstance(key, (pd.Index, abc.KeysView)):
        key = list(key)
    is_list = (
        isinstance(key, (list, tuple, np.record))
        if allow_array
        else isinstance(key, (list, tuple, np.ndarray, np.record))
    )
    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return key if is_list_of_str else key if is_list and len(key) < 20 else [key]


def compute_velocity_on_grid(
    X_emb,
    V_emb,
    density = None,
    smooth = None,
    n_neighbors = None,
    min_mass = None,
    autoscale = True,
    adjust_for_stream = False,
    cutoff_perc = None,
):
    
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None: n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors = n_neighbors, n_jobs = -1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid**2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None: cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)
        V_grid[0][cutoff] = np.nan

    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]
        if autoscale: V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid


def get_components(components = None, basis = None, projection = None):

    if components is None:
        components = "1,2,3" if projection == "3d" else "1,2"
    if isinstance(components, str):
        components = components.split(",")
    components = np.array(components).astype(int) - 1
    if "diffmap" in basis or "vmap" in basis:
        components += 1
    return components


def velocity_embedding_stream(
    adata,
    basis = 'umap',
    vkey = "velocity",
    neighbor_key = 'neighbors',
    density = 2,
    smooth = None,
    min_mass = None,
    cutoff_perc = None,
    arrow_color = None,
    arrow_size = 1,
    arrow_style = "-|>",
    max_length = 4,
    integration_direction = "both",
    linewidth = None,
    n_neighbors = None,
    recompute = None,
    color = 'black',
    layer = None,
    color_map = 'Turbo',
    X = None,
    V = None,
    X_grid = None,
    V_grid = None,
    groups = None,
    components = None,
    figsize = (4, 4),
    dpi = 100,
    ax = None,
    **kwargs,
):
    
    if vkey == "all":
        lkeys = list(adata.layers.keys())
        vkey = [key for key in lkeys if "velocity" in key and "_u" not in key]

    color, color_map = kwargs.pop("c", color), kwargs.pop("cmap", color_map)
    colors = make_unique_list(color, allow_array = True)
    layers, vkeys = make_unique_list(layer), make_unique_list(vkey)

    if V is None:
        for key in vkeys:
            if recompute or (f'{vkey}.{basis}' not in adata.obsm.keys()):
                velocity_embedding(adata, basis = basis, vkey = key, neighbor_key = neighbor_key)

    color, layer, vkey = colors[0], layers[0], vkeys[0]
    if X_grid is None or V_grid is None:
        
        _adata = (
            adata[groups_to_bool(adata, groups, groupby = color)]
            if groups is not None and color in adata.obs.keys()
            else adata
        )

        comps, obsm = get_components(components, basis), _adata.obsm
        X_emb = np.array(obsm[f"{basis}"][:, comps]) if X is None else X[:, :2]
        V_emb = np.array(obsm[f"{vkey}.{basis}"][:, comps]) if V is None else V[:, :2]
        X_grid, V_grid = compute_velocity_on_grid(
            X_emb = X_emb,
            V_emb = V_emb,
            density = 1,
            smooth = smooth,
            min_mass = min_mass,
            n_neighbors = n_neighbors,
            autoscale = False,
            adjust_for_stream = True,
            cutoff_perc = cutoff_perc,
        )

        lengths = np.sqrt((V_grid**2).sum(0))
        linewidth = 1 if linewidth is None else linewidth
        linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()

    stream_kwargs = {
        "linewidth": linewidth,
        "density": density or 2,
        "zorder": 3,
        "arrow_color": arrow_color or "k",
        "arrowsize": arrow_size or 1,
        "arrowstyle": arrow_style or "-|>",
        "maxlength": max_length or 4,
        "integration_direction": integration_direction or "both",
    }

    embedding_kwargs = {
        'ptsize': 8,
        'size': None,
        'hue_order': None,
        'default_color': 'black',
        'alpha': 0.8,
        'rasterize': True,
        'sort': True,
        'annotate': True,
        'annotate_style': 'index',
        'annotate_foreground': 'black',
        'annotate_stroke': 'white',
        'annotate_fontsize': 12,
        'legend': False,
        'contour_plot': False,
        'contour_fill': False,
        'contour_mask': None,
        'contour_mask_values': [],
        'contour_linewidth': 0.8,
        'contour_default_color': 'black',
        'contour_alpha': 1,
        'contour_levels': 10,
        'contour_bw': 0.5,
        'legend_col': 1,
        'add_outline': False,
        'outline_color': 'black',
        'outline_margin': 20,
        'title': None,
        'sample_name': None,
        'cmap_reverse': False,
        'cmap_lower': '#000000',
        'hue_norm': None,
        'legend_loc': 'right margin',
        'frameon': 'small'
    }

    if ax is None: fig, ax = pl.subplots(figsize = figsize, dpi = dpi)
    else: fig = ax.figure

    for arg in list(kwargs):
        if arg in stream_kwargs: stream_kwargs.update({arg: kwargs[arg]})
        if arg in embedding_kwargs: embedding_kwargs.update({arg: kwargs[arg]})

    stream_kwargs["color"] = stream_kwargs.pop("arrow_color", "k")
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)

    embedding(
        adata = adata, basis = basis, color = color, cmap = color_map,
        figsize = figsize, dpi = dpi, ax = ax, **embedding_kwargs
    )
    
    return fig


def velocity(
    adata,
    gene = None,
    basis = "umap",
    vkey = "velocity",
    mode = None,
    fits = None,
    layers = "all",
    color = None,
    color_map = 'Turbo',
    perc = None,
    alpha = 0.5,
    size = None,
    groupby = None,
    groups = None,
    legend_loc = "none",
    fontsize = None,
    figsize = (6, 3),
    dpi = 100,
    ax = None,
    ncols = None,
    neighbor_key: str = 'neighbors',
    highly_variable: str = 'hvg.vst',
    disp_norm: str = 'disp.norm',
    likelihood: str = 'fit.likelihood',
    **kwargs,
):
    if perc is None: perc = [2, 98]
    color, color_map = kwargs.pop("c", color), kwargs.pop("cmap", color_map)
    if fits is None: fits = ["velocity", "dynamics"]
    if color_map is None: color_map = ["RdYlGn", "gnuplot_r"]

    if isinstance(groupby, str) and (groupby in adata.obs.keys()) and (gene is None):
        if "rank.velocity.genes" not in adata.uns.keys() \
            or adata.uns["rank.velocity.genes"]["params"]["groupby"] != groupby:
            rank_velocity_genes(
                adata, vkey = vkey, n_genes = 10, groupby = groupby,
                neighbor_key = neighbor_key, highly_variable = highly_variable,
                disp_norm = disp_norm, likelihood = likelihood
            )
            
        names = np.array(adata.uns["rank.velocity.genes"]["names"].tolist())
        if groups is None: gene = names[:, 0]
        else:
            groups = [groups] if isinstance(groups, str) else groups
            categories = adata.obs[groupby].cat.categories
            idx = np.array([any([g in group for g in groups]) for group in categories])
            gene = np.hstack(names[idx, : int(10 / idx.sum())])
    
    elif gene is not None:
        if isinstance(gene, str): gene = translate_variables(adata, [gene])
        else: 
            gene = translate_variables(adata, gene_list = gene)
            gene = [var for var in gene if var in adata.var_names]
    
    else: error("no gene or groups specified.")
    gene = pd.unique(gene)

    if "ms" not in adata.layers.keys(): skey, ukey = "spliced", "unspliced"
    else: skey, ukey = "ms", "mu"
    layers = [vkey, skey] if layers == "all" else layers
    layers = [layer for layer in layers if layer in adata.layers.keys() or layer == "X"]

    fits = list(adata.layers.keys()) if fits == "all" else fits
    fits = [fit for fit in fits if f"{fit}.gamma" in adata.var.keys()] + ["dynamics"]
    stochastic_fits = [fit for fit in fits if f"variance.{fit}" in adata.layers.keys()]

    nplts = 1 + len(layers) + (mode == "stochastic") * 2
    ncols = 1 if ncols is None else ncols
    nrows = int(np.ceil(len(gene) / ncols))
    ncols = int(ncols * nplts)

    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    if ax is None: ax = pl.figure(figsize = figsize, dpi = dpi)
    gs = pl.GridSpec(nrows, ncols, wspace = 0.25, hspace = 0.4)

    embedding_kwargs = {
        'ptsize': 8,
        'size': None,
        'hue_order': None,
        'default_color': 'black',
        'rasterize': True,
        'sort': True,
        'annotate': True,
        'annotate_style': 'index',
        'annotate_foreground': 'black',
        'annotate_stroke': 'white',
        'annotate_fontsize': 12,
        'legend': False,
        'contour_plot': False,
        'contour_fill': False,
        'contour_mask': None,
        'contour_mask_values': [],
        'contour_linewidth': 0.8,
        'contour_default_color': 'black',
        'contour_alpha': 1,
        'contour_levels': 10,
        'contour_bw': 0.5,
        'legend_col': 1,
        'add_outline': False,
        'outline_color': 'black',
        'outline_margin': 20,
        'sample_name': None,
        'cmap_reverse': False,
        'cmap_lower': '#000000',
        'hue_norm': None,
        'legend_loc': 'right margin',
    }

    scatter_kwargs = {
        "size": size,
        "annotate_fontsize": fontsize
    }

    embedding_kwargs.update(scatter_kwargs)

    for v, var in enumerate(gene):

        _adata = adata[:, var]
        s, u = _adata.layers[skey], _adata.layers[ukey]
        if issparse(s): s, u = s.toarray(), u.toarray()

        # spliced/unspliced phase portrait with steady-state estimate
        ax = pl.subplot(gs[v * nplts])
        cmap = color_map
        if isinstance(color_map, (list, tuple)):
            cmap = color_map[-1] if color in ["X", skey] else color_map[0]
        if "xlabel" not in kwargs: kwargs["xlabel"] = "spliced"
        if "ylabel" not in kwargs: kwargs["ylabel"] = "unspliced"
        legend_loc_lines = "none" if v < len(gene) - 1 else legend_loc

        embedding(
            adata,
            basis = np.array([s.squeeze(), u.squeeze()]).T,
            color = groupby,
            cmap = cmap,
            frameon = 'small',
            # human readable name
            title = adata.var.loc[var, 'gene'],
            alpha = alpha,
            ax = ax,
            xlabel = 'spliced',
            ylabel = 'unspliced',
            ** embedding_kwargs,
        )

        # velocity and expression plots
        for layer_id, layer in enumerate(layers):
            ax = pl.subplot(gs[v * nplts + layer_id + 1])
            title = "expression" if layer in ["X", skey] else layer
            cmap = color_map
            if isinstance(color_map, (list, tuple)):
                cmap = color_map[-1] if layer in ["X", skey] else color_map[0]
            
            embedding(
                adata,
                basis = basis,
                slot = layer,
                color = var,
                cmap = cmap,
                frameon = 'small',
                title = title,
                alpha = alpha,
                ax = ax,
                xlabel = None,
                ylabel = None,
                ** embedding_kwargs
            )

        if mode == "stochastic":
            
            ss, us = second_order_moments(_adata, neighbor_key = neighbor_key)
            s, u, ss, us = s.flatten(), u.flatten(), ss.flatten(), us.flatten()
            fit = stochastic_fits[0]

            ax = pl.subplot(gs[v * nplts + len(layers) + 1])
            beta, offset = 1, 0
            if f"{fit}.beta" in adata.var.keys():
                beta = _adata.var[f"{fit}.beta"]
            if f"{fit}.offset" in adata.var.keys():
                offset = _adata.var[f"{fit}.offset"]
            
            x = np.array(2 * (ss - s**2) - s)
            y = np.array(2 * (us - u * s) + u + 2 * s * offset / beta)
            kwargs["xlabel"] = r"2 $\Sigma_s - \langle s \rangle$"
            kwargs["ylabel"] = r"2 $\Sigma_{us} + \langle u \rangle$"
            
            embedding(
                adata,
                basis = np.array([x, y]).T,
                color = groupby,
                cmap = cmap,
                frameon = 'small',
                title = 'stochastic',
                alpha = alpha,
                ax = ax,
                xlabel = 'spliced',
                ylabel = 'unspliced',
                ** embedding_kwargs
            )

            xnew = np.linspace(np.min(x), np.max(x) * 1.02)
            for fit in stochastic_fits:
                gamma, beta, offset2 = 1, 1, 0
                if f"{fit}.gamma" in adata.var.keys():
                    gamma = _adata.var[f"{fit}.gamma"].values
                if f"{fit}.beta" in adata.var.keys():
                    beta = _adata.var[f"{fit}.beta"].values
                if f"{fit}.offset2" in adata.var.keys():
                    offset2 = _adata.var[f"{fit}.offset2"].values
                ynew = gamma / beta * xnew + offset2 / beta
                pl.plot(xnew, ynew, c = "k", linestyle = "--")

    return ax.figure


def proportions(
    adata,
    groupby = "leiden",
    layers = None,
    highlight = "unspliced",
    add_labels_pie = True,
    add_labels_bar = True,
    fontsize = 8,
    figsize = (10, 2),
    dpi = 100,
    use_raw = True,
):
    
    # get counts per cell for each layer
    if layers is None: layers = ["spliced", "unspliced", "ambiguous"]
    layers_keys = [key for key in layers if key in adata.layers.keys()]
    counts_layers = [sum(adata.layers[key], axis = 1) for key in layers_keys]

    if use_raw:
        ikey, obs = "n.umi.", adata.obs
        counts_layers = [
            obs[ikey + layer_key] if ikey + layer_key in obs.keys() else c
            for layer_key, c in zip(layers_keys, counts_layers)
        ]

    counts_total = np.sum(counts_layers, 0)
    counts_total += counts_total == 0
    counts_layers = np.array([counts / counts_total for counts in counts_layers])

    gspec = pl.GridSpec(1, 2, pl.figure(None, figsize, dpi=dpi))
    colors = pl.get_cmap("tab20b")(np.linspace(0.10, 0.65, len(layers_keys)))

    # pie chart of total abundances
    ax = pl.subplot(gspec[0])
    if highlight is None: highlight = "none"
    explode = [
        0.1 if (layer_key == highlight or layer_key in highlight) else 0
        for layer_key in layers_keys
    ]

    autopct = "%1.0f%%" if add_labels_pie else None
    pie = ax.pie(
        np.mean(counts_layers, axis = 1),
        colors = colors,
        explode = explode,
        autopct = autopct,
        shadow = True,
        startangle = 45,
    )
    if autopct is not None:
        for pct, color in zip(pie[-1], colors):
            r, g, b, _ = color
            pct.set_color("white" if r * g * b < 0.5 else "darkgrey")
            pct.set_fontweight("bold")
            pct.set_fontsize(fontsize)
    
    ax.legend(
        layers_keys,
        ncol=len(layers_keys),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize=fontsize,
    )

    # bar chart of abundances per category
    if groupby is not None and groupby in adata.obs.keys():

        counts_groups = {}
        for cluster in adata.obs[groupby].cat.categories:
            counts_groups[cluster] = np.mean(
                counts_layers[:, adata.obs[groupby] == cluster], axis=1
            )

        labels = list(counts_groups.keys())
        data = np.array(list(counts_groups.values()))
        data_cum = data.cumsum(axis=1)

        ax2 = pl.subplot(gspec[1])
        for i, (colname, color) in enumerate(zip(layers_keys, colors)):
            starts, widths = data_cum[:, i] - data[:, i], data[:, i]
            xpos = starts + widths / 2
            curr_xpos = xpos[0]
            for i, (x, w) in enumerate(zip(xpos, widths)):
                if not (x - w / 2 + 0.05 < curr_xpos < x + w / 2 - 0.05):
                    curr_xpos = x
                xpos[i] = curr_xpos

            ax2.barh(
                labels, widths, left = starts, height = 0.9, 
                label = colname, color = color
            )

            if add_labels_bar:
                r, g, b, _ = color
                text_color = "white" if r * g * b < 0.5 else "darkgrey"
                for y, (x, c) in enumerate(zip(xpos, widths)):
                    ax2.text(
                        x,
                        y,
                        f"{(c * 100):.0f}%",
                        ha = "center",
                        va = "center",
                        color = text_color,
                        fontsize = fontsize,
                        fontweight = "bold",
                    )

        ax2.legend(
            ncol = len(layers_keys),
            bbox_to_anchor = (0, 1),
            loc = "lower left",
            fontsize = fontsize,
        )
        ax2.invert_yaxis()
        ax2.set_xlim(0, np.nansum(data, axis=1).max())
        ax2.margins(0)

        ax2.set_xlabel("proportions", fontweight = "bold", fontsize = fontsize * 1.2)
        ax2.set_ylabel(groupby, fontweight = "bold", fontsize = fontsize * 1.2)
        ax2.tick_params(axis = "both", which = "major", labelsize = fontsize)
        ax = [ax, ax2]
    
    return ax[0].figure
