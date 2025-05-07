
import seaborn as sns
from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
from exprmat.utils import find_variable


def barplot(
    adata, *, gene, 
    slot = 'X', group = 'cell.type', selected_groups = None, 
    split = 'treat', palette = ['red', 'black'], 
    size = (6,3), dpi = 100
):
    
    colnames = ['logc' + x for x in [gene]]
    cellnames = ['c' + str(i) for i in range(len(adata.obs_names.tolist()))]
    lognorm = find_variable(adata, gene, layer = slot)
    data = pd.DataFrame(lognorm.toarray())
    
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

    fig, axes = plt.subplots(1, 1, figsize = size, dpi = dpi)
    plt.rcParams["ytick.labelright"] = True
    plt.rcParams["ytick.labelleft"] = False
    
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
    axes.set_xticklabels(selected_groups, rotation = 45)
    axes.spines[['left', 'top']].set_visible(False)
    axes.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    fig.tight_layout()
    return fig