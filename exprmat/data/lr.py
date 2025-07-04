
from pandas import read_table, isnull
import pandas as pd
from numpy import unique
import numpy as np
import os
import itertools
from pandas import DataFrame

from exprmat.lr.utils import default_params as V
from exprmat.lr.utils import default_primary_columns as P
from exprmat.ansi import error, warning, info
from exprmat.data.finders import basepath


def select_resource(taxa, resource_name: str = V.resource_name) -> DataFrame:
    
    resource_name = resource_name.lower()
    resource_path = os.path.join(basepath, taxa, 'lr', resource_name + '.tsv')

    if not os.path.exists(resource_path):
        error(f"ligand-receptor interaction database name `{taxa}/{resource_name}` not found.")

    resource = read_table(resource_path, sep = '\t', index_col = False)
    resource = resource[['source.gene', 'target.gene']]
    resource = resource.rename(columns = {
        'source.gene': P.ligand,
        'target.gene': P.receptor
    })

    return resource


def handle_resource(
    destination_taxa = 'hsa', source_taxa = 'hsa', interactions = None, resource = None, resource_name = None, 
    x_name = P.ligand, y_name = P.receptor, min_evidence = 3, verbose = True
):
    
    if interactions is None:
        if resource is None:
            if resource_name is None:
                error("if 'interactions' and 'resource' are both None, 'resource_name' must be provided.")
            else: 
                
                resource = select_resource(source_taxa, resource_name)

                # here, we will map the taxa with registered interaction database (source taxa) 
                # to the destination (currently handling) taxa
                if source_taxa != destination_taxa:
                    resource = translate_resource(
                        resource, map_df = get_orthologs(
                            taxa = source_taxa, destination = destination_taxa,
                            min_evidence = min_evidence, columns = None
                        ), columns = ['ligand', 'receptor'], replace = True,

                        # here, we will be harsher and only keep mappings that don't map to more 
                        # than 1 genes as orthologs. these genes may recombine if set > 1.
                        one_to_many = 1
                    )
        
        else:
            if (not isinstance(resource, DataFrame)) or \
                (x_name not in resource.columns) or \
                (y_name not in resource.columns):
                error(
                    "if 'interactions' is None, 'resource' must be a valid data frame "
                    "with columns '{}' and '{}'.".format(x_name, y_name)
                )

            resource = resource.copy()
            resource = resource.dropna(subset = [x_name, y_name]).drop_duplicates()
            resource.index = range(len(resource))
            resource.index.name = None
    else:
        if not isinstance(interactions, list) or any(len(item) != 2 for item in interactions):
            raise ValueError("'interactions' should be a list of tuples in the format [(x1, y1), (x2, y2), ...].")
        interactions = set(interactions)
        resource = DataFrame(interactions, columns = [x_name, y_name])

    return resource


def replace_subunits(lst, my_dict, one_to_many):

    result = []
    for x in lst:
        if x in my_dict:
            value = my_dict[x]
            if not isinstance(value, list): value = [value]
            if len(value) > one_to_many: result.append(np.nan)
            else: result.append(value)
        else: result.append(np.nan)
    return result


def generate_orthologs(data, column, map_dict, one_to_many):

    df = data[[column]].drop_duplicates().set_index(column)
    df["subunits"] = df.index.str.split("_")
    df["subunits"] = df["subunits"].apply(
        replace_subunits,
        args = (map_dict, one_to_many),
    )

    df = df["subunits"].explode().reset_index()
    grouped = df.groupby(column).filter(lambda x: x["subunits"].notna().all()).groupby(column)

    # generate all possible subunit combinations within each group
    complexes = []
    for name, group in grouped:
        if group["subunits"].isnull().all(): continue
        subunit_lists = [list(x) for x in group["subunits"]]
        complex_combinations = list(itertools.product(*subunit_lists))
        for complex in complex_combinations:
            complexes.append((name, "_".join(complex)))

    # Create output DataFrame
    col_names = ["orthology.source", "orthology.target"]
    result = pd.DataFrame(complexes, columns = col_names).set_index("orthology.source")

    return result


def translate_column(resource, map_df, column, replace=True, one_to_many = 1,):
    
    if not isinstance(one_to_many, int):
        error("`one_to_many` should be a positive integer.")
    if ['source', 'target'] != map_df.columns.tolist():
        error("The `map_df` data frame must have two columns named 'source' and 'target'.")

    # get orthologs
    map_df = map_df.set_index("source")
    map_dict = map_df.groupby(level = 0)["target"].apply(list).to_dict()
    map_data = generate_orthologs(resource, column, map_dict, one_to_many)

    # join orthologs
    resource = resource.merge(
        map_data, left_on = column, right_index = True, how = "left")

    # replace orthologs
    if replace: resource[column] = resource["orthology.target"]
    else: resource[f"orthology.{column}"] = resource.apply(
            lambda x: x["orthology.target"]
            if not pd.isnull(x["orthology.target"])
            else x[column],
            axis = 1,
        )

    resource = resource.drop(columns=["orthology.target"])
    resource = resource.dropna(subset=[column])
    return resource


# function that loops over columns and applies translate_column
def translate_resource(resource, map_df, columns = ['ligand', 'receptor'], **kwargs):
    for column in columns:
        resource = translate_column(resource, map_df, column, **kwargs)
    return resource


def get_orthologs(taxa, destination, min_evidence = 3, columns = None ):
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

    filename = os.path.join(basepath, taxa, 'orthologs', destination + '.tsv.gz')
    if not os.path.exists(filename):
        error(f'the orthologs of requested species `{destination}` (from {taxa}) is not registered in database.')
    
    mapping = read_table(filename, sep = "\t")
    mapping['evidence'] = mapping['support'].apply(lambda x: len(x.split(",")))
    mapping = mapping[mapping['evidence'] >= min_evidence]

    if columns is not None:
        mapping = mapping[columns]
    else:
        columns = []
        for x in mapping.columns.tolist():
            if x.endswith('_symbol'): columns.append(x)
        mapping = mapping[columns]

    mapping.columns.names = ['source', 'target']
    return mapping