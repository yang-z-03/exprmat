
import seaborn as sns
from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd

from exprmat.utils import find_variable, choose_layer
from exprmat.ansi import error


def barplot(
    adata, *, gene, 
    slot = 'X', group = 'cell.type', selected_groups = None, selected_splits = None,
    split = 'treat', palette = ['red', 'black'], ax = None, style = 'box',
    size = (6,3), dpi = 100, violin_kwargs = {}
):
    
    plt.rcParams["ytick.labelright"] = False
    plt.rcParams["ytick.labelleft"] = True
    
    colnames = ['logc' + x for x in [gene]]
    cellnames = ['c' + str(i) for i in range(len(adata.obs_names.tolist()))]
    lognorm = find_variable(adata, gene, layer = slot)
    data = pd.DataFrame(lognorm)
    
    data.columns = colnames
    if split == None: data['split'] = 'default.split'
    else: data['split'] = adata.obs[split].tolist()
    data['row'] = adata.obs[group].tolist()
    data['cell'] = cellnames

    longform = pd.wide_to_long(data, 'logc', i = ['cell', 'split', 'row'], j = 'name', suffix = r'\w+')
    longform = longform.reset_index()
    dropzeros = longform

    if selected_groups is None: selected_groups = adata.obs[group].value_counts().index.tolist()
    select_ct = [x in selected_groups for x in dropzeros['row'].tolist()]
    dropzeros['selected.celltype'] = select_ct
    dropzeros = dropzeros.loc[dropzeros['selected.celltype'],:]

    if selected_splits is None:
        if split is not None: selected_splits = adata.obs[split].value_counts().index.tolist()
        else: selected_splits = ['default.split']

    select_sp = [x in selected_splits for x in dropzeros['split'].tolist()]
    dropzeros = dropzeros.loc[select_sp,:]

    if ax is None: fig, axes = plt.subplots(1, 1, figsize = size, dpi = dpi)
    else: axes = ax

    if style == 'box':

        sns.boxplot(
            data = dropzeros.loc[dropzeros['name'] == gene,:],
            x = "row", y = "logc", hue = 'split', ax = axes, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', linecolor = '0', order = selected_groups, gap = 0.3,
            flierprops = dict(markerfacecolor = '0.1', markersize = 1, linestyle = 'none'),
            showfliers = True, width = 0.6
        )
    
    elif style == 'violin':

        sns.violinplot(
            data = dropzeros.loc[dropzeros['name'] == gene,:],
            x = "row", y = "logc", hue = 'split', ax = axes, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', linecolor = '0', order = selected_groups, gap = 0,
            width = 0.8, **violin_kwargs
        )
    
    else: error('can only set plot style to `box` or `violin`.')
    
    axes.tick_params(
        axis = 'y', right = False, left = True, color = 'gray', length = 4,
        grid_color = 'none'
    )
        
    axes.tick_params(
        axis = 'x', left = True, color = 'black', length = 4,
        grid_color = 'lightgray'
    )
            
    axes.legend(bbox_to_anchor = (1, 1), loc = 'upper left', borderaxespad = 0, frameon = False)
    axes.set_xlabel('')
    axes.set_ylabel(gene)
    # axes.set_xticks(selected_groups)
    axes.set_yticks([0.0, 2.5])

    # axes.set_xticklabels(selected_groups, rotation = 45)

    from scipy import stats
    # here, we will run the wilcox test
    gene_data = dropzeros.loc[dropzeros['name'] == gene,:]
    if group != None:
        xtl = []
        for ct in selected_groups:
            cell_data = gene_data.loc[gene_data['row'] == ct,:]
            groups = [x for _, x in cell_data.groupby(cell_data['split'])]
            values = [x['logc'].tolist() for x in groups]
            if len(values) == 2:
                w = stats.mannwhitneyu(values[0], values[1])
                xtl += [ct + '\n{:.3f}'.format(w.pvalue)]
            else: xtl += [ct]
        
        # axes.set_xticks(selected_groups)
        axes.set_xticklabels(xtl, rotation = 45)
    
    else:
        groups = [x for _, x in gene_data.groupby(gene_data['split'])]
        values = [x['logc'].tolist() for x in groups]
        xtl = 'whole'
        if len(values) == 2:
            w = stats.mannwhitneyu(values[0], values[1])
            xtl = 'whole\n{:.3f}'.format(w.pvalue)
        axes.set_xticklabels([xtl], rotation = 45)
        pass

    axes.spines[['right', 'top']].set_visible(False)
    axes.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    if ax is None: return fig
    else: return axes.figure


def compare_scatter(
    adata, *, group_x, group_y, key, slot = 'X', markers = [],
    ax = None, sample = None, figsize = (4, 4), dpi = 100
):
    import numpy as np
    x = choose_layer(adata[adata.obs[key] == group_x, :], layer = slot).copy()
    y = choose_layer(adata[adata.obs[key] == group_y, :], layer = slot).copy()
    mean_x = np.array(x.mean(axis = 0))[0]
    mean_y = np.array(y.mean(axis = 0))[0]
    varn = adata.var['gene'].tolist()

    show_text = True
    if isinstance(markers, str):
        import exprmat.reader.static as st
        
        df1 = st.rna_get_markers(
            adata, None, de_slot = markers, max_q = 0.05, min_pct = 0, max_pct_reference = 1,
            min_lfc = -100, max_lfc = -0.5
        )

        df2 = st.rna_get_markers(
            adata, None, de_slot = markers, max_q = 0.05, min_pct = 0, max_pct_reference = 1,
            min_lfc = 0.5, max_lfc = 100
        )

        markers = df1['gene'].tolist() + df2['gene'].tolist()
        show_text = False
    
    ann = ['annot' if x in markers else 'bg' for x in varn]

    if ax is None: fig, axes = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
    else: axes = ax

    sns.scatterplot(
        x = mean_x, y = mean_y, ax = ax, hue = ann,
        hue_order = ['bg', 'annot'], palette = ['gray', 'red'],
        s = 4, edgecolor = None, legend = False,
        rasterized = True
    )

    plt.xlabel(f'Average expression in {group_x}')
    plt.ylabel(f'Average expression in {group_y}')
    plt.plot([0, mean_x.max()], [0, mean_x.max()], linewidth = 1, linestyle = '--', color = 'red')

    for x, y, name in zip(mean_x, mean_y, varn):
        if name in markers:
            if show_text: plt.text(x = x, y = y, s = name, fontsize = 9, color = 'black')

    if ax is None: return fig
    else: return axes.figure


def matrix(
    adata, layer = 'X', obs_names = None, var_names = None, 
    figsize = (3, 3), ax = None, **kwargs
):
    mat = choose_layer(adata, layer = layer)
    
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    else: fig = ax.figure

    ax.imshow(mat, **kwargs)

    obs = adata.obs_names.tolist() if obs_names is None else adata.obs[obs_names].tolist()
    var = adata.var_names.tolist() if var_names is None else adata.obs[var_names].tolist()
    ax.set_xticks(range(len(var)), labels = var, rotation = 90, ha = "right", va = 'center', rotation_mode = "anchor")
    ax.set_yticks(range(len(obs)), labels = obs)

    return fig


def volcano(
    adata, de_slot = 'deg', label = [],
    show_all = False, min_pct = 0, max_pct_reference = 1, 
    min_lfc = -25, max_lfc = 25, remove_zero_pval = False,
    highlight_min_logp = 5, highlight_min_lfc = 1.5,
    xlim = 5, ylim = 100,
    figsize = (3, 3), dpi = 100
):

    import numpy as np
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import exprmat.reader.static as st

    pvals = []
    fc = []
    names = []

    if isinstance(de_slot, str):
        plots = st.rna_get_markers(
            adata, None, de_slot = de_slot,
            min_pct = min_pct, max_pct_reference = max_pct_reference, 
            min_lfc = min_lfc, max_lfc = max_lfc, 
            remove_zero_pval = remove_zero_pval
        )
        
    else: plots = de_slot
    
    pvals = plots['log10.q'].tolist()
    fc = plots['lfc'].tolist()
    names = plots['gene'].tolist()

    fig, axes = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
    axes.scatter(fc, pvals, c = 'k', s = 4)

    sig_pvals = []
    sig_fc = []
    sig_names = []

    for p, f, nm in zip(pvals, fc, names):
        if nm in label:
            sig_pvals += [p]
            sig_fc += [f]
            sig_names += [nm]

        if (p > highlight_min_logp and abs(f) >= highlight_min_lfc) and show_all:
            sig_pvals += [p]
            sig_fc += [f]
            sig_names += [nm]

    texts = []
    for p, f, nm in zip(sig_pvals, sig_fc, sig_names):
        texts += [axes.text(f, p, nm, va = 'top', ha = 'center')]

    # adjust_text(texts)
    axes.scatter(sig_fc, sig_pvals, c = 'r', s = 10)
    axes.grid(False)

    xlim = min(np.abs(np.array(fc)).max(), xlim)
    ylim = min(ylim, np.max(np.array(pvals)))
    axes.set_xlim([-xlim, xlim])
    axes.set_ylim([-ylim / 20, ylim])

    axes.vlines(x = [0], ymin = [0], ymax = [ylim], linestyles = 'dashed', color = 'gray', linewidth = 0.6)
    axes.set_yticks([])
    axes.set_xlabel('Log-2 fold changes')
    for pos in ['right', 'top', 'left']:
        axes.spines[pos].set_visible(False)
        
    return fig