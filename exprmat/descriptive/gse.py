
import numpy as np
from exprmat.data.geneset import get_genesets, translate_id
from exprmat.ansi import info, error, red, green


def gse(
    adata, *, taxa,

    # differential expression slots:
    de_slot, group_name = None,
    min_pct = 0.0, max_pct_reference = 1, 
    min_lfc = None, max_lfc = None, remove_zero_pval = False,

    key_added = 'gsea',
    gene_sets = 'all',
    identifier = 'entrez'
):
    
    params = adata.uns[de_slot]['params']

    # default value for convenience
    if len(adata.uns[de_slot]['differential']) == 1 and group_name == None:
        group_name = list(adata.uns[de_slot]['differential'].keys())[0]

    info('fetched diff `' + red(group_name) + '` over `' + green(params['reference']) + '`')
    tab = adata.uns[de_slot]['differential'][group_name]

    if min_pct is not None and 'pct' in tab.columns:
        tab = tab[tab['pct'] >= min_pct]
    if max_pct_reference is not None and 'pct.reference' in tab.columns:
        tab = tab[tab['pct.reference'] <= max_pct_reference]
    if min_lfc is not None and 'lfc' in tab.columns:
        tab = tab[tab['lfc'] >= min_lfc]
    if max_lfc is not None and 'lfc' in tab.columns:
        tab = tab[tab['lfc'] <= max_lfc]
    if remove_zero_pval:
        tab = tab[~ np.isinf(tab['log10.q'].to_numpy())]
    
    tab = tab.sort_values(by = ['lfc'], ascending = False)
    input_gnames = tab.loc[:, ['names', 'lfc']].copy()

    names = input_gnames['names'].tolist()
    names = [x.replace('rna:', '') for x in names]

    import gseapy as gp
    import pandas as pd

    prernk = pd.DataFrame({
        'gene': translate_id(taxa, names, 'ugene', identifier, keep_nones = True),
        'lfc': input_gnames['lfc'].tolist()
    })

    prernk = prernk.loc[~ prernk['gene'].duplicated(), :].copy()
    prernk = prernk.loc[~ prernk['gene'].isna(), :].copy()
    gsea = gp.prerank(
        rnk = prernk,
        gene_sets = get_genesets(taxa = taxa, name = gene_sets, identifier = identifier)
    )

    adata.uns[key_added] = {
        'config': {
            'module': gsea.module,
            'ranking.key': np.array(gsea.ranking.index.tolist()),
            'ranking.value': np.array(gsea.ranking.values),
            'seed': gsea.seed,
            'negative': gsea.pheno_neg,
            'positive': gsea.pheno_pos
        },
        'results': {}
    }

    for k in gsea.results.keys():
        adata.uns[key_added]['results'][k] = {
            'name': gsea.results[k]['name'],
            'es': gsea.results[k]['es'],
            'res': np.array(gsea.results[k]['RES']),
            'nes': gsea.results[k]['nes'],
            'p': gsea.results[k]['pval'],
            'fdr': gsea.results[k]['fdr'],
            'fwerp': gsea.results[k]['fwerp'],
            'tag': gsea.results[k]['tag %'],
            'lead': np.array(gsea.results[k]['lead_genes'].split(';'), dtype = str),
            'match': np.array(gsea.results[k]['matched_genes'].split(';'), dtype = 'str'),
            'hits': np.array(gsea.results[k]['hits'])
        }
    
    pass