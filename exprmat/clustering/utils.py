
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import random


class igraph_rng:

    def __init__(self, random_state: int | np.random.RandomState = 0) -> None:
        self._rng = check_random_state(random_state)

    def __getattr__(self, attr: str):
        return getattr(self._rng, "normal" if attr == "gauss" else attr)
    

def set_igraph_random_state(random_state):

    import igraph
    rng = igraph_rng(random_state)
    try: igraph.set_random_number_generator(rng); yield None
    finally: igraph.set_random_number_generator(random)


def rename_groups(
    adata: ad.AnnData,
    restrict_key: str,
    *,
    key_added: str | None,
    restrict_categories,
    restrict_indices,
    groups
):

    key_added = f"{restrict_key}_R" if key_added is None else key_added
    all_groups = adata.obs[restrict_key].astype("U")
    prefix = "-".join(restrict_categories) + ","
    new_groups = [prefix + g for g in groups.astype("U")]
    all_groups.iloc[restrict_indices] = new_groups
    return all_groups


def restrict_adjacency(
    adata: ad.AnnData,
    restrict_key: str,
    *,
    restrict_categories,
    adjacency
):
    if not isinstance(restrict_categories[0], str):
        msg = "You need to use strings to label categories, e.g. '1' instead of 1."
        raise ValueError(msg)
    for c in restrict_categories:
        if c not in adata.obs[restrict_key].cat.categories:
            msg = f"{c!r} is not a valid category for {restrict_key!r}"
            raise ValueError(msg)
    restrict_indices = adata.obs[restrict_key].isin(restrict_categories).values
    adjacency = adjacency[restrict_indices, :]
    adjacency = adjacency[:, restrict_indices]
    return adjacency, restrict_indices