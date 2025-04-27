
import os
import pandas
from exprmat.configuration import default as cfg

basepath = os.path.dirname(__file__)
genome = {}

def refresh_genome():
    '''
    You should set the configuration prior to any possible call to this package (by default Mus 
    musculus). If you change the target taxa, you should manually call ``refresh_genome`` to 
    force an update of the data.
    '''

    genome = {}
    return


def get_genome(taxa):
    '''
    Retrieve the gene annotation table for specified species. Note that the species settings is
    automatically read from the configuration. You should set the configuration prior to any
    possible call to this package (by default Mus musculus). If you change the target taxa, you
    should manually call ``refresh_genome`` to force an update of the data.
    '''

    if taxa not in genome.keys(): genome[taxa] = {}
    if 'genes' in genome[taxa].keys(): return genome[taxa]['genes']
    genome[taxa]['genes'] = pandas.read_table(
        os.path.join(basepath, f'{taxa}', 'genome.tsv.gz'),
        # set low memory to false to allow correct adjustment to mixed dtype.
        index_col = '.ugene', dtype = {'.seqid': str}
    )

    return genome[taxa]['genes']


def get_mapper_name(taxa):
    '''
    Return a name mapper mapping from gene names to universal gene id across species.
    This list will be automatically updated when it experiences more name-accession pairs when
    reading the 10x features table. This aims to adapt to the changing nomenclature in different
    versions of cellranger annotations.
    '''

    # we will construct the gene table for ensembl id and gene name alias at once during
    # the startup process of this script. then we will be able to directly calls for
    # its ugene id in all the next times.

    if taxa not in genome.keys(): genome[taxa] = {}
    if 'mapper.name' in genome[taxa].keys(): return genome[taxa]['mapper.name']
    import pickle
    if os.path.exists(os.path.join(basepath, f'{taxa}', 'mapping-name.pkl')):

        with open(os.path.join(basepath, f'{taxa}', 'mapping-name.pkl'), 'rb') as fens:
            name_finder = pickle.load(fens)

    else:

        gtable = get_genome(taxa)
        name_list = gtable[['gene']].values.transpose()[0].tolist()
        name_finder = {}

        # alias_list = gtable[['alias']].values.transpose()[0].tolist()
        id_list = gtable.index.tolist()
        duplicates = []
        alias_finder = {}

        # for x in range(len(alias_list)):
        #     alias = alias_list[x]
        #     if not isinstance(alias, str): continue
        #     if len(alias) > 0:
        #         spl = alias.split(';')
        #         for y in spl:
        #             if y not in alias_finder.keys(): alias_finder[y] = x
        #             elif y not in duplicates: duplicates += [y]

        for x in duplicates: del alias_finder[x]
        for k in alias_finder.keys(): name_finder[k] = taxa + ':' + id_list[alias_finder[k]]

        for i in range(len(name_list)):
            name_finder[name_list[i]] = taxa + ':' + id_list[i]

        # write the generated name mapping to file cache.
        with open(os.path.join(basepath, f'{taxa}', 'mapping-name.pkl'), 'wb') as f:
            pickle.dump(name_finder, f)

    genome[taxa]['mapper.name'] = name_finder
    return name_finder


def get_mapper_ensembl(taxa):
    '''
    Return a name mapper mapping from ENSEMBL gene IDs to universal gene id across species.
    This list will be automatically updated when it experiences more name-accession pairs when
    reading the 10x features table. This aims to adapt to the changing nomenclature in different
    versions of cellranger annotations.
    '''

    if taxa not in genome.keys(): genome[taxa] = {}
    if 'mapper.ensembl' in genome[taxa].keys(): return genome[taxa]['mapper.ensembl']
    import pickle
    if os.path.exists(os.path.join(basepath, f'{taxa}', 'mapping-ensembl.pkl')):

        with open(os.path.join(basepath, f'{taxa}', 'mapping-ensembl.pkl'), 'rb') as fens:
            ensembl_finder = pickle.load(fens)

    else:

        gtable = get_genome(taxa)
        ensembl_list = gtable[['ensembl']].values.transpose()[0].tolist()
        id_list = gtable.index.tolist()
        ensembl_finder = {}

        for i in range(len(ensembl_list)):
            ensembl_finder[ensembl_list[i]] = taxa + ':' + id_list[i]
    
        # write the generated name mapping to file cache.
        with open(os.path.join(basepath, f'{taxa}', 'mapping-ensembl.pkl'), 'wb') as f:
            pickle.dump(ensembl_finder, f)

    genome[taxa]['mapper.ensembl'] = ensembl_finder
    return ensembl_finder


def update_mapper_name(taxa, name, ugene):
    '''
    Update the gene name to unique gene id mapping.
    '''
    
    if taxa not in genome.keys():
        get_mapper_name(taxa)

    if 'mapper.name' not in genome[taxa].keys():
        get_mapper_name(taxa)
    
    if name not in genome[taxa]['mapper.name'].keys():
        genome[taxa]['mapper.name'][name] = ugene
    
    return


def update_mapper_ensembl(taxa, ensembl, ugene):
    '''
    Update the gene ENSEMBL ID to unique gene id mapping.
    '''

    if taxa not in genome.keys():
        get_mapper_ensembl(taxa)

    if 'mapper.ensembl' not in genome[taxa].keys():
        get_mapper_ensembl(taxa)
    
    if ensembl not in genome[taxa]['mapper.ensembl'].keys():
        genome[taxa]['mapper.ensembl'][ensembl] = ugene
    
    return


def save_genome_changes(taxa):
    '''
    Write the changes to the mapper to disk. Remember to save this if you would like to memorize
    the new mappings learned from your data to faciliate later use with this mapping.
    '''

    name_finder = get_mapper_name(taxa)
    ensembl_finder = get_mapper_ensembl(taxa)
    import pickle

    with open(os.path.join(basepath, f'{taxa}', 'mapping-name.pkl'), 'wb') as f:
        pickle.dump(name_finder, f)
    with open(os.path.join(basepath, f'{taxa}', 'mapping-ensembl.pkl'), 'wb') as f:
        pickle.dump(ensembl_finder, f)
    
    return