
import os
import re
from enum import Enum, unique
from typing import Optional, Tuple, Type, Union, Set
from abc import ABCMeta, abstractmethod
from cytoolz import memoize

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as pf

from exprmat.data.signature import signature
from exprmat.ansi import error, warning


@unique
class region_type(Enum):
    REGIONS = "regions"
    GENES = "genes"


@unique
class cistrome_type(Enum):
    MOTIFS = "motifs"
    TRACKS = "tracks"


@unique
class cistarget_db_format(Enum):
    SCORES = "scores"
    RANKINGS = "rankings"


class region_index:

    @staticmethod
    def get_regions_from_bed(
        bed_filename: str,
        gene_regex: Optional[str] = None,
    ):
        gene_ids = list()
        region_ids = list()
        gene_ids_set = set()
        region_ids_set = set()

        with open(bed_filename, mode = "r", encoding = "utf-8") as fh:
            for line in fh:
                if line and not line.startswith("#"):
                    columns = line.strip().split("\t")

                    if len(columns) < 4:
                        error('bed files must contain at least 4 columns.')

                    # region index from column 4 of the bed file.
                    region_id = columns[3]

                    if gene_regex:
                        gene_id = re.sub(gene_regex, "", region_id)
                        if gene_id not in gene_ids_set:
                            gene_ids.append(gene_id)
                            gene_ids_set.add(gene_id)
                    
                    else:
                        # if not matching genes, each region should have unique names,
                        # genes are commonly notated as prefixes to regions.
                        if region_id in region_ids_set:
                            error('region names should be unique.')
                        else:
                            region_ids.append(region_id)
                            region_ids_set.add(region_id)

        if gene_regex: return region_index(
            region_or_gene_ids = gene_ids,
            regions_or_genes_type = region_type.GENES)
        
        else: return region_index(
            region_or_gene_ids = region_ids,
            regions_or_genes_type = region_type.REGIONS)


    @staticmethod
    def get_regions_from_fasta(
        fasta_filename: str,
        gene_regex: Optional[str] = None,
    ):

        gene_ids = list()
        region_ids = list()
        gene_ids_set = set()
        region_ids_set = set()

        with open(fasta_filename, mode = "r", encoding = "utf-8") as fh:
            for line in fh:
                if line.startswith(">"):
                    region_id = line[1:].split(maxsplit = 1)[0]

                    if gene_regex:
                        gene_id = re.sub(gene_regex, "", region_id)
                        if gene_id not in gene_ids_set:
                            gene_ids.append(gene_id)
                            gene_ids_set.add(gene_id)
                    
                    else:
                        if region_id in region_ids:
                            error('region names should be unique.')
                        else:
                            region_ids.append(region_id)
                            region_ids_set.add(region_id)


        if gene_regex: return region_index(
            region_or_gene_ids = gene_ids,
            regions_or_genes_type = region_type.GENES)
        
        else: return region_index(
            region_or_gene_ids = region_ids,
            regions_or_genes_type = region_type.REGIONS)


    def __init__(
        self, region_or_gene_ids,
        regions_or_genes_type,
    ):
        
        if isinstance(region_or_gene_ids, set):
            region_or_gene_ids = sorted(region_or_gene_ids)

        # collapse duplicates, keep order, and add sort value.
        self.ids_dict = {rg_id: idx for idx, rg_id in enumerate(region_or_gene_ids)}

        if len(region_or_gene_ids) != len(self.ids_dict):
            # recreate dict if region id or gene id's contained duplicates.
            self.ids_dict = {rg_id: idx for idx, rg_id in enumerate(self.ids_dict)}

        self.ids = tuple(self.ids_dict)
        self.ids_set = set(self.ids_dict)
        self.type = region_type(regions_or_genes_type)


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, region_index): return NotImplemented
        return self.type == other.type and self.ids_set == other.ids_set

    def __len__(self) -> int: return len(self.ids)

    def __getitem__(self, items):
        if isinstance(items, int): return region_index((self.ids[items],), self.type)
        return region_index(self.ids[items], self.type)


    def difference(self, other):
        
        if not isinstance(other, region_index): return NotImplemented
        assert self.type == other.type, "cannot compare regions or genes index that is not of the same type"
        return region_index(
            region_or_gene_ids = sorted(
                self.ids_set.difference(other.ids_set), key = lambda x: self.ids_dict[x]),
            regions_or_genes_type = self.type
        )


    def intersection(self, other):
        
        if not isinstance(other, region_index): return NotImplemented
        assert self.type == other.type, "cannot compare regions or genes index that is not of the same type"
        return region_index(
            region_or_gene_ids = sorted(
                self.ids_set.intersection(other.ids_set), key = lambda x: self.ids_dict[x]),
            regions_or_genes_type = self.type,
        )


    def issubset(self, other) -> bool:
        """
        Check if all region or gene IDs in the current object are at least 
        present in the other object.
        """
        if not isinstance(other, region_index): return NotImplemented
        assert self.type == other.type, "cannot compare regions or genes index that is not of the same type"
        return self.ids_set.issubset(other.ids_set)


    def issuperset(self, other) -> bool:
        """
        Check if all region or gene IDs in the other object are at least 
        present in the current object.
        """
        if not isinstance(other, region_index): return NotImplemented
        assert self.type == other.type, "cannot compare regions or genes index that is not of the same type"
        return self.ids_set.issuperset(other.ids_set)
    

    def sort(self):
        return region_index(sorted(self.ids), self.type)


    def union(self, other):

        if not isinstance(other, region_index): return NotImplemented
        assert self.type == other.type, "cannot compare regions or genes index that is not of the same type"
        return region_index(
            region_or_gene_ids = sorted(
                self.ids_set.union(other.ids_set),
                regions_or_genes_type = self.type,
                key = lambda x: (
                    self.ids_dict.get(x, len(self.ids) + 1),
                    other.ids_dict.get(x, 0),
                )
            )
        )


    def is_genes(self) -> bool:
        return self.type == region_type.GENES

    def is_regions(self) -> bool:
        return self.type == region_type.REGIONS


class cistrome_index:
    """
    MotifOrTrackIDs class represents a unique sorted tuple of motif IDs or track IDs 
    for constructing a Pndas dataframe index for a cisTarget database.
    """

    def __init__(
        self, motif_or_track_ids,
        motifs_or_tracks_type,
    ):

        if isinstance(motif_or_track_ids, set):
            motif_or_track_ids = sorted(motif_or_track_ids)

        # collapse duplicates, keep order, and add sort value.
        self.ids_dict = {rg_id: idx for idx, rg_id in enumerate(motif_or_track_ids)}

        if len(motif_or_track_ids) != len(self.ids_dict):
            # recreate dict if motif IDs or track IDs contained duplicates.
            self.ids_dict = {mt_id: idx for idx, mt_id in enumerate(self.ids_dict)}

        self.ids = tuple(self.ids_dict)
        self.ids_set = set(self.ids_dict)
        self.type = cistrome_type(motifs_or_tracks_type)


    def __eq__(self, other: object):
        if not isinstance(other, cistrome_index): return NotImplemented
        return self.type == other.type and self.ids_set == other.ids_set

    def __len__(self): return len(self.ids)

    def __getitem__(self, items):
        if isinstance(items, int): return cistrome_index((self.ids[items],), self.type)
        return cistrome_index(self.ids[items], self.type)

    def sort(self):
        return cistrome_index(sorted(self.ids), self.type)

    def is_motifs(self):
        return self.type == cistrome_type.MOTIFS

    def is_tracks(self):
        return self.type == cistrome_type.TRACKS


class cistarget_db:
    """
    CisTargetDatabase class for reading rankings or scores for regions or genes from a 
    cis-target scores or rankings database. The database is a ranking or score table
    (commonly stored in feather or anndata format for disk space efficiency).
    """

    @staticmethod
    def init_ct_db(
        ct_db_filename,
        expected_score_or_ranking = 'rankings',
        expected_row_type = 'motifs',
        expected_column_type = 'genes',
        engine = 'pyarrow',
    ):
        # get column names from the feather file with pyarrow without loading the whole database.
        # the lazy loading is implemented in prefetch.
        schema = ds.dataset(ct_db_filename, format = "feather").schema
        column_names = schema.names
        dtypes = schema.types

        index_column_idx: Optional[int] = None
        index_column_name: Optional[str] = None

        # get database index column ("motifs", "tracks", "regions" or "genes" depending of the database type).
        # Start with the last column (as the index column normally should be the latest).
        for column_idx, column_name in zip(
            range(len(column_names) - 1, -1, -1), column_names[::-1]
        ):
            if column_name in {"motifs", "tracks", "regions", "genes"}:
                index_column_idx = column_idx
                index_column_name = column_name

                row_names = (
                    pf.read_table(
                        source = ct_db_filename,
                        columns = [column_idx],
                        memory_map = False,
                        use_threads = False,
                    )
                    .column(0)
                    .to_pylist()
                )

        if not index_column_name or not index_column_idx:
            warning(f'no columns named "motifs", "tracks", "regions" or "genes" in the data frame.')
            error(f'{ct_db_filename} is not a cistarget feather database.')

        # get all column names without index column name.
        column_names = (
            column_names[0:index_column_idx] + column_names[index_column_idx + 1 :]
        )

        # get dtype for those columns (should be the same for all of them).
        column_dtype = list(
            set(dtypes[0:index_column_idx] + 
                dtypes[index_column_idx + 1 :])
        )

        if len(column_dtype) != 1:
            error(f"only one dtype is allowed for {column_names[0:3]} ...: {column_dtype}")

        column_dtype = column_dtype[0]
        dtype: Union[Type[np.int16], Type[np.int32], Type[np.float32]]

        # infer content from data type.
        if column_dtype == pa.int16():
            scores_or_rankings = "rankings"
            dtype = np.int16
        elif column_dtype == pa.int32():
            scores_or_rankings = "rankings"
            dtype = np.int32
        elif column_dtype == pa.float32():
            scores_or_rankings = "scores"
            dtype = np.float32
        else: error(f'unsupported dtype "{column_dtype}" for cistarget database.')
        
        # get cistarget database type from cisTarget database filename extension.
        
        row_kind = index_column_name
        if expected_score_or_ranking != scores_or_rankings:
            error('unexpected dtype from your specified content mode.')
        if expected_row_type != row_kind:
            error('unexpected row type (motifs or tracks) from your content and assertions.')

        # Assume column kind is correct if the other values were correct.
        column_kind = expected_column_type

        if column_kind in ("regions", "genes"):
            # Create cisTarget database object if the correct database was provided.
            return cistarget_db(

                ct_db_filename = ct_db_filename,

                region_or_gene_ids = region_index(
                    region_or_gene_ids = column_names,
                    regions_or_genes_type = column_kind,
                ),

                motif_or_track_ids = cistrome_index(
                    motif_or_track_ids = row_names,
                    motifs_or_tracks_type = row_kind
                ),

                scores_or_rankings = scores_or_rankings,
                dtype = dtype,
                engine = engine,
            )
        
        else: error('invalid cistarget database.')


    def __init__(
        self, ct_db_filename,
        region_or_gene_ids: region_index,
        motif_or_track_ids: cistrome_index,
        scores_or_rankings: cistarget_db_format,
        dtype, engine = 'pyarrow'
    ):
        
        # cisTarget scores or rankings database file.
        self.ct_db_filename = ct_db_filename

        self.all_region_or_gene_ids: region_index = region_or_gene_ids
        self.all_motif_or_track_ids: cistrome_index = motif_or_track_ids
        self.scores_or_rankings: cistarget_db_format = scores_or_rankings
        self.dtype: Type[Union[np.int16, np.int32, np.float32]] = dtype
        self.engine = engine

        # count number of region IDs or gene IDs.
        self.n_regions = len(self.all_region_or_gene_ids)
        # count number of motif IDs or track IDs.
        self.n_cistromes = len(self.all_motif_or_track_ids)

        # pyarrow table with scores or rankings for those region IDs or gene IDs that where loaded
        # with prefetch(). this acts as a cache.
        self.df_cached = None

        # keep track for which region ids or gene ids, scores or rankings are
        # loaded with prefetch().
        self.region_or_gene_ids_loaded: Optional[region_index] = None


    @property
    def is_row_genes(self) -> bool:
        return self.all_region_or_gene_ids.type == region_type.GENES

    @property
    def is_row_regions(self) -> bool:
        return self.all_region_or_gene_ids.type == region_type.REGIONS

    @property
    def is_column_motifs(self) -> bool:
        return self.all_motif_or_track_ids.type == cistrome_type.MOTIFS

    @property
    def is_column_tracks(self) -> bool:
        return self.all_motif_or_track_ids.type == cistrome_type.TRACKS

    @property
    def is_scores(self) -> bool:
        return self.scores_or_rankings == cistarget_db_format.SCORES

    @property
    def is_rankings(self) -> bool:
        return self.scores_or_rankings == cistarget_db_format.RANKINGS
    

    def is_regions_found(
        self, region_or_gene_ids: region_index
    ):
        """ 
        Check if all input regions or genes are found in the database. 
        
        Returns
        -------
        - A boolean indicates whether all requested features are found.
        - Found features in index.
        - Unfound features in index.
        """
        
        if region_or_gene_ids.issubset(self.all_region_or_gene_ids): return (
            True, region_or_gene_ids, region_index([], region_or_gene_ids.type))
        else: return (
            False, region_or_gene_ids.intersection(self.all_region_or_gene_ids),
            region_or_gene_ids.difference(self.all_region_or_gene_ids)
        )


    def clear_cache(self):

        self.df_cached = None
        self.region_or_gene_ids_loaded = None


    def prefetch_pyarrow(
        self, region_or_gene_ids: region_index, sort=False
    ) -> None:
        
        have_all, found_regions, notfound_regions = self.is_regions_found(region_or_gene_ids)
        if have_all is False: error('you requested for regions that do not exist.')

        if not self.df_cached or not isinstance(self.df_cached, pa.Table):
            self.df_cached = pf.read_table(
                source = self.ct_db_filename,
                columns = (
                    found_regions.sort().ids
                    if sort else found_regions.ids
                ) + (self.all_motif_or_track_ids.type.value,),
                memory_map = False,
                use_threads = True,
            )

            self.region_or_gene_ids_loaded = found_regions
        
        else:
            # get region IDs or gene IDs subset for which no scores/rankings were loaded before.
            region_or_gene_ids_to_load = found_regions.difference(
                self.region_or_gene_ids_loaded)

            # check if new region IDs or gene IDs need to be loaded.
            if len(region_or_gene_ids_to_load) != 0:
                # get region IDs or gene IDs subset columns with scores/rankings 
                # from cistarget feather file as a pyarrow Table.
                pa_table_subset = pf.read_table(
                    source = self.ct_db_filename,
                    columns = region_or_gene_ids_to_load.ids,
                    memory_map = False,
                    use_threads = True,
                )

                # get current loaded pyarrow table.
                pa_table = self.df_cached

                for column in pa_table_subset.itercolumns():
                    # Append column with region IDs or gene IDs scores/rankings to existing pyarrow Table.
                    pa_table = pa_table.append_column(column._name, column)

                # Keep track of loaded region IDs or gene IDs scores/rankings.
                self.region_or_gene_ids_loaded = found_regions.union(
                    self.region_or_gene_ids_loaded
                )

                # Store new pyarrow Table with previously and newly loaded region IDs or gene IDs scores/rankings.
                self.df_cached = pa_table.select((
                        self.region_or_gene_ids_loaded.sort().ids
                        if sort else self.region_or_gene_ids_loaded.ids
                    ) + (self.all_motif_or_track_ids.type.value,)
                )


    def prefetch(
        self,
        region_or_gene_ids: region_index,
        engine = None,
        sort: bool = False,
    ):
        have_all, found_regions, nfound_regions = self.is_regions_found(region_or_gene_ids)
        if have_all is False: error('you requested for regions that do not exist.')

        engine = engine if engine else self.engine
        self.prefetch_pyarrow(region_or_gene_ids = region_or_gene_ids, sort = sort)


    def subset_to_pandas(
        self, region_or_gene_ids: region_index, engine = None,
    ):
        have_all, found_regions, nfound_regions = self.is_regions_found(region_or_gene_ids)
        if have_all is False: error('you requested for regions that do not exist.')
        engine = engine if engine else self.engine

        # fetch scores or rankings for input region IDs or gene IDs from cistarget database 
        # file for region IDs or gene IDs which were not prefetched in previous calls.
        self.prefetch(region_or_gene_ids = region_or_gene_ids, engine = engine, sort = True)
        if not self.df_cached: error('prefetch failed.')

        pd_df = pd.DataFrame(
            data = self.df_cached.select(
                found_regions.ids
                + (("motifs",) if self.is_column_motifs else ("tracks",))
            ).to_pandas()
        )

        # set motifs or tracks column as index (inplace to avoid extra copy).
        pd_df.set_index("motifs" if self.is_column_motifs else "tracks", inplace = True)

        # add "regions" or "genes" as column index name.
        pd_df.rename_axis(
            columns = "regions" if self.is_row_regions else "genes", inplace = True
        )

        return pd_df



class ranking_db(metaclass = ABCMeta):
    """
    A class of a database of whole genome rankings. The whole genome is ranked for
    regulatory features of interest, e.g. motifs for a transcription factor.
    The rankings of the genes are 0-based.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def n_genes(self) -> int:
        pass

    @property
    @abstractmethod
    def genes(self) -> Tuple[str]:
        pass

    @property
    @memoize
    def geneset(self) -> Set[str]:
        return set(self.genes)

    @abstractmethod
    def load_full(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load(self, gs: signature) -> pd.DataFrame:
        pass

    def __str__(self):
        return self.name


class ranking_db_feather(ranking_db):

    def __init__(
        self, fname: str, name: str, 
        expected_score_or_ranking: str = 'rankings',
        expected_row_type: str = 'motifs',
        expected_column_type: str = 'genes'
    ):
        super().__init__(name = name)
        if not os.path.isfile(fname): error(f'invalid file attempt to be loaded. ({fname})')

        self._fname = fname
        self.ct_db = cistarget_db.init_ct_db(
            ct_db_filename = self._fname, engine = "pyarrow",
            expected_column_type = expected_column_type,
            expected_row_type = expected_row_type,
            expected_score_or_ranking = expected_score_or_ranking
        )


    @property
    @memoize
    def n_genes(self) -> int:
        return self.ct_db.n_regions


    @property
    @memoize
    def genes(self) -> Tuple[str]:
        return self.ct_db.all_region_or_gene_ids.ids


    def load_full(self) -> pd.DataFrame:
        return self.ct_db.subset_to_pandas(
            region_or_gene_ids = self.ct_db.all_region_or_gene_ids
        )


    def load(self, gs: signature) -> pd.DataFrame:
        # for some genes in the signature there might not be a rank 
        # available in the database.
        gene_set = self.geneset.intersection(set(gs.genes))
        return self.ct_db.subset_to_pandas(
            region_or_gene_ids = region_index(
                region_or_gene_ids = gene_set,
                regions_or_genes_type = self.ct_db.all_region_or_gene_ids.type,
            )
        )


class inmemory(ranking_db):
    """
    A decorator for a ranking database which loads the entire database in memory.
    """

    def __init__(self, db: ranking_db):
        assert db, "database should be supplied."
        self._db = db
        self._df = db.load_full()
        super().__init__(db.name)

    @property
    def n_genes(self) -> int:
        return self._db.n_genes

    @property
    def genes(self) -> Tuple[str]:
        return self._db.genes

    def load_full(self) -> pd.DataFrame:
        return self._df

    def load(self, gs: signature) -> pd.DataFrame:
        return self._df.loc[:, self._df.columns.isin(gs.genes)]


def opendb(
    fname: str, name: str, 
    expected_score_or_ranking: str = 'rankings',
    expected_row_type: str = 'motifs',
    expected_column_type: str = 'genes'
) -> ranking_db:

    extension = os.path.splitext(fname)[1]
    if extension == ".feather":
        return ranking_db_feather(
            fname, name = name, 
            expected_column_type = expected_column_type,
            expected_row_type = expected_row_type,
            expected_score_or_ranking = expected_score_or_ranking
        )
    
    else: error('invalid database format.')
