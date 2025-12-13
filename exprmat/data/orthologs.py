
from pandas import read_feather, DataFrame
import os
from exprmat.ansi import error, warning, info
from exprmat.data.finders import basepath


def get_orthologs(taxa, destination):
    """
    HCOP is a composite database combining data from various orthology resources.
    It provides a comprehensive set of orthologs among human, mouse, and rat, among many other species.

    If you use this function, please reference the original HCOP papers:
    -  Eyre, T.A., Wright, M.W., Lush, M.J. and Bruford, E.A., 2007. HCOP: a searchable database of 
       human orthology predictions. Briefings in bioinformatics, 8(1), pp.2-5.
    -  Yates, B., Gray, K.A., Jones, T.E. and Bruford, E.A., 2021. Updates to HCOP: the HGNC comparison 
       of orthology predictions tool. Briefings in Bioinformatics, 22(6), p.bbab155.

    For more information, please visit the HCOP website: https://www.genenames.org/tools/hcop/,
    or alternatively check the bulk download FTP links page: https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/
    """

    # query and expand full table
    ortho = read_feather(os.path.join(basepath, 'shared', 'orthologs.feather'))
    if (taxa not in ortho.columns) or (destination not in ortho.columns):
        error(f'the orthologs of requested species `{destination}` (from {taxa}) is not registered in database.')
    
    table = {
        'source.entrez': [],
        'source.ensembl': [],
        'source.nom': [],
        'source': [],

        'dest.entrez': [],
        'dest.ensembl': [],
        'dest.nom': [],
        'dest': []
    }

    selection = ortho[[
        f'{taxa}.entrez',
        f'{taxa}.ensembl',
        f'{taxa}.nom',
        taxa,
        f'{destination}.entrez',
        f'{destination}.ensembl',
        f'{destination}.nom',
        destination
    ]].copy()

    selection = selection[
        (selection[f'{taxa}.ensembl'] != '-') & 
        (selection[f'{destination}.ensembl'] != '-')
    ]

    for i, row in selection.iterrows():
        s_entrez = row.iloc[0].split('; ')
        s_ensembl = row.iloc[1].split('; ')
        s_nom = row.iloc[2].split('; ')
        s_sym = row.iloc[3].split('; ')

        t_entrez = row.iloc[4].split('; ')
        t_ensembl = row.iloc[5].split('; ')
        t_nom = row.iloc[6].split('; ')
        t_sym = row.iloc[7].split('; ')

        for s1, s2, s3, s4 in zip(s_entrez, s_ensembl, s_nom, s_sym):
            for t1, t2, t3, t4 in zip(t_entrez, t_ensembl, t_nom, t_sym):
                
                table['source.entrez'].append(s1)
                table['source.ensembl'].append(s2)
                table['source.nom'].append(s3)
                table['source'].append(s4)

                table['dest.entrez'].append(t1)
                table['dest.ensembl'].append(t2)
                table['dest.nom'].append(t3)
                table['dest'].append(t4)
    
    expanded = DataFrame(table)
    expanded = expanded.loc[~expanded[['source.ensembl', 'dest.ensembl']].duplicated(), :].copy()
    expanded = expanded[
        (expanded[f'source.ensembl'] != '-') & 
        (expanded[f'dest.ensembl'] != '-')
    ]
    
    return expanded


def get_orthologs_symbol(taxa, destination, suppress_one_to_many = True, suppress_many_to_one = True):

    orthologs = get_orthologs(taxa, destination)
    orthologs['source'] = [y if x == '-' else x for x, y in zip(orthologs['source'], orthologs['source.ensembl'])]
    orthologs['dest'] = [y if x == '-' else x for x, y in zip(orthologs['dest'], orthologs['dest.ensembl'])]

    # one source can only represent one dest
    orthologs = orthologs[['source', 'dest']].copy()

    if suppress_one_to_many:
        indices = orthologs['source'].loc[orthologs['source'].duplicated()].tolist()
        orthologs = orthologs.loc[[x not in indices for x in orthologs['source']], :].copy()
    
    if suppress_many_to_one:
        indices = orthologs['dest'].loc[orthologs['dest'].duplicated()].tolist()
        orthologs = orthologs.loc[[x not in indices for x in orthologs['dest']], :].copy()

    return orthologs
