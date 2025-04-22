
import os
import pandas


class metadata:
    """
    Metadata at sample level. This class is a wrapper around sample level metadata table in 
    pandas format, and can be dumped and loaded using plain text tables.
    
    Parameters
    ----------

    locations : list[str]
        File system paths indicating directories where the 10x-like matrix files locate. The 
        file reading scheme is default to 10x matrix files including ``matrix.mtx``, 
        ``barcodes.tsv`` and ``features.tsv``. Appropriate compression levels and column 
        information is guessed from file extensions and content columns of ``features.tsv`` 
        (or ``genes.tsv``).
    
    names : list[str] | None
        Sample names in the loaded experiment. If left to None, names of the folder will be 
        used. Since the folder names may duplicate, this is not forced to be unique. This might 
        introduce some unexpected settings, so you are recommended to set this parameter explicitly.
    
    batches : list[str] | None
        Batch information, if not set, this column is set to uniform value '.'.
    
    groups : list[str] | None
        Experimental groupings, if not set, this column is set to uniform value '.'.
    
    Notes
    -----------
    The inner metadata table follows several naming conventions. That auto-generated tables
    must have column ``location`` and ``sample`` for locations and sample names, and ``batch``
    and ``group`` columns for batches and experimental groupings, and ``modality`` for library
    modality, ``taxa`` for default taxa where no prefix in features is specified. other metadata 
    information is not necessary, and can be appended as specified by user, but do not set duplicate
    column names as these six.
    """

    def __init__(
            self, locations, modality, default_taxa,
            names = None, batches = None, groups = None, df = None
        ):
        
        if df is not None:
            if isinstance(df, pandas.DataFrame):
                self.dataframe = df
                return

        n_sample = len(locations)
        
        # generate sample names from common prefix.
        if names is not None:
            assert len(names) == n_sample
        else: 
            prefix = os.path.commonprefix(locations)
            names = [
                x.replace(prefix, '').replace('/', '.').replace('\\', '.').lower() \
                for x in locations
            ]
        
        # batches
        if batches is not None:
            assert len(batches) == n_sample
        else: batches = ['.'] * n_sample

        # experimental groupings
        if groups is not None:
            assert len(groups) == n_sample
        else: groups = ['.'] * n_sample

        self.dataframe = pandas.DataFrame({
            'location': locations,
            'sample': names,
            'batch': batches,
            'group': groups,
            'modality': modality,
            'taxa': default_taxa
        })
    

    def save(self, fpath):
        '''
        Write the metadata object into a disk table file.
        '''
        self.dataframe.to_csv(fpath, sep = '\t', index = False, header = True)


    def define_column(self, name, default):
        '''
        Create a new column in the metadata table with given column name. And fill the column with
        the default value. The values can be further defined using conditions of finer scope.
        We recommend indicating all conditions with the carefully named sample names, and use
        simple conditional filters to process the sample names.

        It should be noted that the column names should not be duplicated with the pre-defined ones:
        one of ``location``, ``modality``, ``sample``, ``batch``, ``taxa`` and ``group``, unless you 
        are sure about what you are doing.
        '''
        self.dataframe[name] = default

    
    def set_paste(self, key1, key2, dest, sep = ':'):
        '''
        Paste the stringify values of two keys with separator
        '''
        
        source1 = self.dataframe[key1].tolist()
        source2 = self.dataframe[key2].tolist()
        paste = [str(s1) + sep + str(s2) for s1, s2 in zip(source1, source2)]
        self.dataframe[dest] = paste
    

    def set_fraction(self, key = 'sample', dest = 'group', sep = '.', fraction = 0, fallback = '.'):
        '''
        Alter the content of a column if starting with a string in the key column.

        Parameters
        ----------

        key : str
            The key column to match the conditional pattern. This column must exist.
        
        dest : str
            The column you may want to alter value according to the patterns in column ``key``, 
            according to the conditions given.
        
        sep : str
            split the key column with specified separator, and picks out the certain fraction by
            sequential index to become the value of metadata.
        
        fallback : str
            If the key column has bad format, what should be used to fill the metadata.
        '''

        source = self.dataframe[key].tolist()
        frac = []
        for t in source:
            if (sep in t) and len(t.split(sep)) > fraction:
                frac += [t.split(sep)[fraction]]
            else: frac += 'fallback'
        self.dataframe[dest] = frac

    
    def set_if_starts(self, key = 'sample', dest = 'group', starts = '.', value = '.'):
        '''
        Alter the content of a column if starting with a string in the key column.

        Parameters
        ----------

        key : str
            The key column to match the conditional pattern. This column must exist.
        
        dest : str
            The column you may want to alter value according to the patterns in column ``key``, 
            according to the conditions given.
        
        starts : str
            Test if values in the ``key`` starts with this string
        
        value : str
            If the condition is true, set the ``dest`` column with this value.
        '''

        source = self.dataframe[key].tolist()
        temp = self.dataframe[dest].tolist() if dest in self.dataframe.columns.tolist() else ['.'] * len(source)
        temp = [value if str(x).startswith(starts) else y for x, y in zip(source, temp)]
        self.dataframe[dest] = temp


    def set_if_ends(self, key = 'sample', dest = 'group', ends = '.', value = '.'):
        '''
        Alter the content of a column if starting with a string in the key column.

        Parameters
        ----------

        key : str
            The key column to match the conditional pattern. This column must exist.
        
        dest : str
            The column you may want to alter value according to the patterns in column ``key``, 
            according to the conditions given.
        
        ends : str
            Test if values in the ``key`` ends with this string
        
        value : str
            If the condition is true, set the ``dest`` column with this value.
        '''

        source = self.dataframe[key].tolist()
        temp = self.dataframe[dest].tolist() if dest in self.dataframe.columns.tolist() else ['.'] * len(source)
        temp = [value if str(x).endswith(ends) else y for x, y in zip(source, temp)]
        self.dataframe[dest] = temp


    def set_if_contains(self, key = 'sample', dest = 'group', contains = '.', value = '.'):
        '''
        Alter the content of a column if starting with a string in the key column.

        Parameters
        ----------

        key : str
            The key column to match the conditional pattern. This column must exist.
        
        dest : str
            The column you may want to alter value according to the patterns in column ``key``, 
            according to the conditions given.
        
        ends : str
            Test if values is contained in the ``key``. This requires that both variables be string.
        
        value : str
            If the condition is true, set the ``dest`` column with this value.
        '''

        source = self.dataframe[key].tolist()
        temp = self.dataframe[dest].tolist() if dest in self.dataframe.columns.tolist() else ['.'] * len(source)
        temp = [value if (contains in str(x)) else y for x, y in zip(source, temp)]
        self.dataframe[dest] = temp

    
def load_metadata(fpath):
    '''
    Read the metadata table from disk.
    '''

    df = pandas.read_table(fpath, sep = '\t', index_col = None)
    assert 'location' in df.columns
    assert 'sample' in df.columns
    assert 'batch' in df.columns
    assert 'group' in df.columns
    assert 'modality' in df.columns
    assert 'taxa' in df.columns
    return metadata(locations = None, modality = None, default_taxa = None, df = df)

