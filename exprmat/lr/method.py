
from __future__ import annotations
import anndata as an
from mudata import MuData
from pandas import DataFrame, concat
from typing import Optional
import weakref

from exprmat.lr.pipeline import pipeline
from exprmat.lr.utils import default_params as V, default_anndata_keys as K
from exprmat.ansi import error, warning, info


class method_config:
    
    # initiate a list to store weak references to all instances
    instances = []

    def __init__(
        self, method_name: str, complex_cols: list, add_cols: list,
        fun, magnitude: str | None, magnitude_ascending: bool | None,
        specificity: str | None, specificity_ascending: bool | None,
        permute: bool, reference: str
    ):
        self.__class__.instances.append(weakref.proxy(self))
        self.method_name = method_name
        self.complex_cols = complex_cols
        self.add_cols = add_cols
        self.fun = fun
        self.magnitude = magnitude
        self.magnitude_ascending = magnitude_ascending
        self.specificity = specificity
        self.specificity_ascending = specificity_ascending
        self.permute = permute
        self.reference = reference


    def by_sample(
        self, adata: an.AnnData | MuData,
        sample_key: str, key_added: str = K.uns_key,
        inplace: bool = V.inplace, verbose = V.verbose, **kwargs
    ):

        if sample_key not in adata.obs:
            error(f"{sample_key} was not found in `adata.obs`.")

        if not adata.obs[sample_key].dtype.name == "category":
            warning(f"converting `{sample_key}` to categorical.")
            adata.obs[sample_key] = adata.obs[sample_key].astype("category")

        verbose = True
        full_verbose = verbose

        samples = adata.obs[sample_key].cat.categories
        adata.uns[key_added] = {}

        for sample in samples:
            info(f'running for sample [{sample}] ...')
            temp = adata[adata.obs[sample_key] == sample]
            if temp.isbacked: temp = temp.to_memory().copy()
            else: temp = temp.copy()

            sample_res = self.__call__(temp, inplace=False, verbose=full_verbose, **kwargs)

            adata.uns[key_added][sample] = sample_res

        liana_res = concat(adata.uns[key_added]).reset_index(level = 1, drop = True).reset_index()
        liana_res = liana_res.rename({"index" : sample_key}, axis = 1)

        if inplace: adata.uns[key_added] = liana_res
        return None if inplace else liana_res


class lr_method(method_config):
    
    def __init__(self, _method):

        super().__init__(
            method_name = _method.method_name,
            complex_cols = _method.complex_cols,
            add_cols = _method.add_cols,
            fun = _method.fun,
            magnitude = _method.magnitude,
            magnitude_ascending = _method.magnitude_ascending,
            specificity = _method.specificity,
            specificity_ascending = _method.specificity_ascending,
            permute = _method.permute,
            reference = _method.reference
        )

        self._method = _method

    
    def __call__(
        self, adata: an.AnnData | MuData,
        taxa_source: str,
        taxa_dest: str,
        groupby: str,
        resource_name: str = V.resource_name,
        expr_prop: float = V.expr_prop,
        min_cells: int = V.min_cells,
        groupby_pairs: Optional[DataFrame] = V.groupby_pairs,
        base: float = V.logbase,
        supp_columns: list = V.supp_columns,
        return_all_lrs: bool = V.return_all_lrs,
        key_added: str = K.uns_key,
        use_raw: Optional[bool] = V.use_raw,
        layer: Optional[str] = V.layer,
        de_method: str = V.de_method,
        n_perms: int = V.n_perms,
        seed: int = V.seed,
        n_jobs: int = 1,
        resource: Optional[DataFrame] = V.resource,
        interactions: Optional[list] = V.interactions,
        mdata_kwargs: dict = dict(),
        inplace: bool = V.inplace,
        verbose: Optional[bool] = V.verbose,
    ):
        
        if supp_columns is None: supp_columns = []
        liana_res = pipeline(
            taxa_source = taxa_source,
            taxa_dest = taxa_dest,
            adata = adata,
            groupby = groupby,
            resource_name = resource_name,
            resource = resource,
            interactions = interactions,
            expr_prop = expr_prop,
            min_cells = min_cells,
            supp_columns = supp_columns,
            return_all_lrs = return_all_lrs,
            groupby_pairs = groupby_pairs,
            base = base,
            de_method = de_method,
            verbose = verbose,
            method_meta = self._method,
            n_perms = n_perms,
            seed = seed,
            n_jobs = n_jobs,
            use_raw = use_raw,
            layer = layer,
            mdata_kwargs = mdata_kwargs
        )
        
        if inplace: adata.uns[key_added] = liana_res
        return None if inplace else liana_res
