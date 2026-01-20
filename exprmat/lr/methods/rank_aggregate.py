
from __future__ import annotations
from exprmat.lr.method import method_config
from exprmat.lr.pipeline import pipeline

from exprmat.lr.utils import (
    default_anndata_keys as K,
    default_params as V
)

import anndata as an
from mudata import MuData
from pandas import DataFrame
from typing import Optional


class aggregate_method(method_config):


    def __init__(self, scoring_methods, methods):

        super().__init__(
            method_name = scoring_methods.method_name,
            complex_cols = [],
            add_cols = [],
            fun = scoring_methods.fun,
            magnitude = scoring_methods.magnitude,
            magnitude_ascending = True,
            specificity = scoring_methods.specificity,
            specificity_ascending = True,
            permute = scoring_methods.permute,
            reference = scoring_methods.reference
        )

        self.scoring_methods = scoring_methods
        self.methods = methods

        self.specificity_specs = { method.method_name: (
            method.specificity, method.specificity_ascending) for method in methods
            if method.specificity is not None
        }

        self.magnitude_specs = { method.method_name: (
            method.magnitude, method.magnitude_ascending) for method in methods
            if method.magnitude is not None
        }

        self.add_cols = list(
            {x for li in [method.add_cols for method in methods] for x in li}
        )

        self.complex_cols = list(
            {x for li in [method.complex_cols for method in methods] for x in li}
        )


    def __call__(
        self, adata: an.AnnData | MuData, groupby: str,
        taxa_source: str,
        taxa_dest: str,
        resource_name: str = V.resource_name,
        expr_prop: float = V.expr_prop,
        min_cells: int = V.min_cells,
        groupby_pairs: Optional[DataFrame] = V.groupby_pairs,
        base: float = V.logbase,
        aggregate_method: str = 'rra',
        consensus_opts: Optional[list] = None,
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
        
        liana_res = pipeline(
            adata = adata,
            groupby = groupby,
            taxa_source = taxa_source, 
            taxa_dest = taxa_dest,
            resource_name = resource_name,
            resource = resource,
            groupby_pairs = groupby_pairs,
            interactions = interactions,
            expr_prop = expr_prop,
            min_cells = min_cells,
            base = base,
            return_all_lrs = return_all_lrs,
            de_method = de_method,
            verbose = verbose,
            method_meta = self,
            use_raw = use_raw,
            layer = layer,
            n_perms = n_perms,
            seed = seed,
            n_jobs = n_jobs,
            consensus_methods = self.methods,
            aggregate_method = aggregate_method,
            consensus_opts = consensus_opts,
            mdata_kwargs = mdata_kwargs
        )

        if inplace: adata.uns[key_added] = liana_res
        return None if inplace else liana_res


ra_config = method_config(
    method_name = "ra",
    complex_cols = [],
    add_cols = [],
    fun = None,
    magnitude = 'magnitude',
    magnitude_ascending = True,
    specificity = 'specificity',
    specificity_ascending = True,
    permute = False,
    reference = 'Nature Communications, 13(1), pp.1-13'
)