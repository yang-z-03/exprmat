
import seaborn as sns
from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
from exprmat.utils import find_variable, choose_layer


def barplot(
    adata, *, gene, 
    slot = 'X', group = 'cell.type', selected_groups = None, 
    split = 'treat', palette = ['red', 'black'], ax = None,
    size = (6,3), dpi = 100
):
    
    plt.rcParams["ytick.labelright"] = True
    plt.rcParams["ytick.labelleft"] = False
    
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

    if ax is None: fig, axes = plt.subplots(1, 1, figsize = size, dpi = dpi)
    else: axes = ax

    sns.boxplot(
        data = dropzeros.loc[dropzeros['name'] == gene,:],
        x = "row", y = "logc", hue = 'split', ax = axes, fill = True,
        palette = palette, saturation = 0.8, log_scale = False,
        color = '.8', linecolor = '0', order = selected_groups, gap = 0.3,
        flierprops = dict(markerfacecolor = '0.1', markersize = 1, linestyle = 'none'),
        showfliers = True, width = 0.6
    )
    
    axes.tick_params(
        axis = 'y', right = True, left = False, color = 'gray', length = 4,
        grid_color = 'none'
    )
        
    axes.tick_params(
        axis = 'x', left = True, color = 'black', length = 4,
        grid_color = 'lightgray'
    )
            
    axes.legend().remove()
    axes.set_xlabel('')
    axes.set_ylabel(gene)
    axes.set_xticks(selected_groups)
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
        
        axes.set_xticks(selected_groups)
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

    axes.spines[['left', 'top']].set_visible(False)
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
    ann = ['annot' if x in markers else 'bg' for x in varn]

    if ax is None: fig, axes = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
    else: axes = ax

    sns.scatterplot(
        x = mean_x, y = mean_y, ax = ax, hue = ann,
        hue_order = ['bg', 'annot'], palette = ['gray', 'red'],
        s = 4, edgecolor = None, legend = False
    )

    plt.xlabel(f'Average expression in {group_x}')
    plt.ylabel(f'Average expression in {group_y}')

    for x, y, name in zip(mean_x, mean_y, varn):
        if name in markers:
            plt.text(x = x, y = y, s = name, fontsize = 9, color = 'black')

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