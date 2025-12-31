
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from exprmat import error


def plot_inferred_ligands(adata, key_nichenet = 'nichenet', figsize = (12, 4), width_ratios = [1, 8]):
    res = adata.uns[key_nichenet]
    if not 'ligand_targets' in res:
        error('for ligand inference, you should run `nichenet_infer_targets` first.')
    
    ltl = res['ligand_targets']
    pvt = pd.pivot_table(ltl, values = "weight", index = "ligand", columns = "target", fill_value = 0)
    fig, axes = plt.subplots(1, 2, sharey = True, width_ratios = width_ratios, figsize = figsize)

    sns.heatmap(
        res['activities'].set_index('ligand').loc[pvt.index, ['aupr.adj']], 
        cmap = 'Oranges', linewidths = 0.8, xticklabels = False, yticklabels = 'auto', ax = axes[0],
        cbar = False
    )

    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation = 0)
    sns.heatmap(
        pvt, cmap = 'Blues', linewidths = 0.8, xticklabels = 1, yticklabels = 'auto',
        ax = axes[1], cbar = False
    )
    return fig