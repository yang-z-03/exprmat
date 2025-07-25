
import numpy as np
import anndata as ad

from exprmat.data.geneset import get_genesets, translate_id
from exprmat.ansi import info, error, red, green
from exprmat.utils import choose_layer


def gse(
    adata, *, taxa,

    # differential expression slots:
    de_slot, group_name = None,
    min_pct = 0.0, max_pct_reference = 1, 
    min_lfc = None, max_lfc = None, remove_zero_pval = False,
    max_q = 0.05,

    key_added = 'gsea',
    gene_sets = 'all',
    identifier = 'entrez'
):
    
    import pandas as pd
    if isinstance(de_slot, pd.DataFrame):
        assert 'names' in de_slot.columns
        assert 'lfc' in de_slot.columns
        assert 'log10.q' in de_slot.columns
        tab = de_slot

    else:    

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
    if max_q is not None and 'q' in tab.columns:
        tab = tab[tab['q'] <= max_q]
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
    info(f'fetched {len(prernk)} preranked genes by logfc.')

    gsea = gp.prerank(
        rnk = prernk,
        gene_sets = get_genesets(taxa = taxa, name = gene_sets, identifier = identifier),
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

    # too big if accepting all gsea results
    for k in gsea.results.keys():

        if (gsea.results[k]['pval'] > 0.1) and \
            (gsea.results[k]['fdr'] > 0.1) and \
            (gsea.results[k]['fwerp'] > 0.1):
            continue 

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


def opa(
    adata, *, taxa,

    # differential expression slots:
    de_slot, group_name = None, translate_de_slot = None,
    min_pct = 0.0, max_pct_reference = 1, 
    min_lfc = None, max_lfc = None, remove_zero_pval = False,
    use_abs_lfc = False, min_abs_lfc = 1.0, max_abs_lfc = 25.0,
    max_q = 0.05,

    key_added = 'opa',
    gene_sets = 'kegg',
    identifier = 'entrez',
    opa_cutoff = 0.05,
):
    
    if isinstance(de_slot, list):
        genes = [x.replace('rna:', '') for x in de_slot]
        if translate_de_slot is not None:
            genes = translate_id(taxa, genes, translate_de_slot, identifier, keep_nones = False)

    else:
        
        import pandas as pd
        if isinstance(de_slot, pd.DataFrame):
            assert 'names' in de_slot.columns
            assert 'lfc' in de_slot.columns
            assert 'log10.q' in de_slot.columns
            tab = de_slot
        
        else:

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
        if max_q is not None and 'q' in tab.columns:
            tab = tab[tab['q'] <= max_q]
        
        if not use_abs_lfc:
            if min_lfc is not None and 'lfc' in tab.columns:
                tab = tab[tab['lfc'] >= min_lfc]
            if max_lfc is not None and 'lfc' in tab.columns:
                tab = tab[tab['lfc'] <= max_lfc]
        else:
            tab = tab[np.abs(tab['lfc'].to_numpy()) >= min_abs_lfc]
            tab = tab[np.abs(tab['lfc'].to_numpy()) <= max_abs_lfc]

        if remove_zero_pval:
            tab = tab[~ np.isinf(tab['log10.q'].to_numpy())]
        
        tab = tab.sort_values(by = ['lfc'], ascending = False)
        input_gnames = tab.loc[:, ['names', 'lfc']].copy()
    
        names = input_gnames['names'].tolist()
        names = [x.replace('rna:', '') for x in names]
        genes = translate_id(taxa, names, 'ugene', identifier, keep_nones = False)

    geneset = get_genesets(taxa = taxa, name = gene_sets, identifier = identifier)
    background = []
    for k in geneset.keys(): background += geneset[k]

    import gseapy as gp
    import pandas as pd

    genes = list(set(genes))
    background = list(set(background))

    info(f'fetched {len(genes)} genes differentially expressed.')
    info(f'with a background of {len(background)} observed genes.')

    op = gp.enrich(
        gene_list = genes,
        gene_sets = geneset,
        background = background,
        cutoff = opa_cutoff
    )
    
    # we prefer to convert the gene names back for human readable results:
    hrgenes = []
    for generep in op.results['Genes'].tolist():
        destids = generep.split(';')
        sourcenames = translate_id(taxa, destids, identifier, 'gene', keep_nones = False)
        hrgenes.append(';'.join(sourcenames))

    adata.uns[key_added] = {
        'gset': op.results['Gene_set'].tolist(),
        'term': op.results['Term'].tolist(),
        'overlap': op.results['Overlap'].tolist(),
        'p': op.results['P-value'].tolist(),
        'fdr': op.results['Adjusted P-value'].tolist(),
        'or': op.results['Odds Ratio'].tolist(),
        'score': op.results['Combined Score'].tolist(),
        'genes': hrgenes,
    }


def gsva(
    adata, taxa, identifier = 'uppercase', gene_sets = 'kegg', lognorm = 'X',
    n_cores = 1, kcdf = 'Gaussian', weight = 1, min_genes = 15, max_genes = 1000
):
    mat = choose_layer(adata, layer = lognorm)
    genes = adata.var_names.tolist()
    genes = [x.replace('rna:', '') for x in genes]
    genes = translate_id(taxa, genes, 'ugene', identifier, keep_nones = True)
    
    # inp = ad.AnnData(X = mat)
    # inp.obs_names = adata.obs_names
    # inp.var_names = genes

    import pandas as pd
    import scipy.sparse as sp

    if sp.issparse(mat): mat = mat.todense()
    inp = pd.DataFrame(mat.T, columns = adata.obs_names.tolist(), index = genes)
    inp = inp.loc[~ inp.index.isna(), :]
    inp = inp.loc[~ inp.index.duplicated(), :].copy()

    import gseapy as gp
    gsv = gp.gsva(
        data = inp, gene_sets = get_genesets(taxa = taxa, name = gene_sets, identifier = identifier),
        kcdf = kcdf, weight = weight, min_size = min_genes, max_size = max_genes, seed = 42,
        threads = n_cores
    )

    pivot = gsv.res2d.pivot(index = 'Name', columns = 'Term', values = 'ES')
    result = ad.AnnData(pivot)
    result = result[adata.obs_names.tolist(), :].copy()
    result.obs = adata.obs
    result.obsm = adata.obsm

    return result