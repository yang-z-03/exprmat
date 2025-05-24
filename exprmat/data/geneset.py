
import os
import pandas
import numpy as np

from exprmat.ansi import warning, info
from exprmat.configuration import default as cfg
from exprmat.data.finders import basepath, genome, get_genome


def get_genesets(taxa, name, identifier = 'entrez'):

    if taxa not in genome:
        genome[taxa] = {}
    
    if 'genesets' not in genome[taxa]:
        genome[taxa]['genesets'] = {}
    
    if name in genome[taxa]['genesets']:
        return genome[taxa]['genesets'][name]
    
    # otherwise, read the gmt file.
    # by default (as in the convention of msigdb, the greatest source of gene
    # set database), the items are recorded in entrez ids.

    gmt = os.path.join(basepath, taxa, 'genesets', name + '.gmt')
    with open(gmt, 'r') as gfile:
        lines = gfile.read().splitlines()
    
    gene_set = {}
    for line in lines:
        
        if len(line.strip()) == 0: continue
        if line.startswith('#'): continue
        tokens = line.split('\t')
        filtered = []
        for t in tokens:
            if len(t.strip()) > 0: filtered.append(t)

        # parse gene set
        gs_name = filtered[0]
        gs_desc = filtered[1]
        gs_genes = filtered[2:]

        if len(gs_genes) > 0:
            gene_set[gs_name] = list(gs_genes)
    
    genome[taxa]['genesets'][name] = gene_set
    return gene_set


def translate_id(taxa, genes, idfrom = 'entrez', idto = 'ugene', keep_nones = True):

    if not taxa in genome.keys():
        get_genome(taxa)

    # setup the conversion table cache
    if not 'conversion' in genome[taxa].keys():

        # construct multi-platform convertion table.
        gtable = get_genome(taxa)
        gtable = gtable[['ensembl', 'gene']].copy()
        gtable['ugene'] = gtable.index.tolist()
        gtable.index = gtable['ensembl'].tolist()
        gtable['uppercase'] = gtable['gene'].str.upper().tolist()

        # deduplicate
        gtable = gtable.loc[~ gtable['ensembl'].duplicated(), :].copy()
        gtable['ugene'] = taxa + ':' + gtable['ugene']

        entrez = pandas.read_table(
            os.path.join(basepath, taxa, 'entrez.tsv.gz'),
            sep = '\t', header = None, index_col = 0, dtype = str
        )

        entrez = entrez.rename(columns = { 1: 'entrez', 2: 'taxa'})
        gtable = gtable.join(entrez, how = 'left')
        genome[taxa]['conversion'] = gtable

    else: gtable = genome[taxa]['conversion']

    froms = np.array(gtable[idfrom].tolist())
    tos = np.array(gtable[idto].tolist())

    if not keep_nones:
        indices = np.where(froms == np.array(genes)[:, None])[-1]
        return tos[indices].tolist()
    
    else:
        result = []
        froms = gtable[idfrom].tolist()
        tos = gtable[idto].tolist()
        for x in genes:
            if x in froms:
                t = tos[froms.index(x)]
                result.append(None if str(t) == 'nan' else t)
            else: result.append(None)
        return result

