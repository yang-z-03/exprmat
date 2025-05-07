
import numpy as np
from exprmat.ansi import error


def kde(x: np.ndarray, y: np.ndarray):
    
    from scipy.stats import gaussian_kde

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    min_z = np.min(z)
    max_z = np.max(z)

    # Scale between 0 and 1
    scaled_z = (z - min_z) / (max_z - min_z)

    return scaled_z


def density(
    adata, basis: str = "umap",
    *,
    groupby: str | None = None,
    key_added: str | None = None,
    components: str = "1,2",
) -> None:
    """
    Calculate the density of cells in an embedding (per condition).

    Gaussian kernel density estimation is used to calculate the density of cells in an embedded 
    space. This can be performed per category over a categorical cell annotation. The cell 
    density can be plotted using the `pl.embedding_density` function.

    Note that density values are scaled to be between 0 and 1. Thus, the density value at each 
    cell is only comparable to densities in the same category. Beware that the KDE estimate 
    used (`scipy.stats.gaussian_kde`) becomes unreliable if you don't have enough cells in a category.

    Parameters
    ----------
    adata
        The annotated data matrix.

    basis
        The embedding over which the density will be calculated. This embedded
        representation is found in `adata.obsm[basis]``.

    groupby
        Key for categorical observation/cell annotation for which densities
        are calculated per category.

    key_added
        Name of the `.obs` covariate that will be added with the density estimates.
    
    components
        The embedding dimensions over which the density should be calculated.
        This is limited to two components.
    """

    # Test user inputs
    basis = basis.lower()

    if basis not in adata.obsm_keys():
        error(f'can not find basis `{basis}`.')

    if components is None:
        components = "1,2"
    if isinstance(components, str):
        components = components.split(",")
    components = np.array(components).astype(int) - 1

    if len(components) != 2:
        error("only support two dimensional kernel density. by default use dimension 1,2")

    if groupby is not None:
        if groupby not in adata.obs:
            msg = f"Could not find {groupby!r} `.obs` column."
            raise ValueError(msg)

        if adata.obs[groupby].dtype.name != "category":
            msg = f"{groupby!r} column does not contain categorical data"
            raise ValueError(msg)

    # define new covariate name
    if key_added is not None:
        density_covariate = key_added
    elif groupby is not None:
        density_covariate = f"density.{basis}.{groupby}"
    else: density_covariate = f"density.{basis}"

    # calculate the densities over each category in the groupby column
    if groupby is not None:
        categories = adata.obs[groupby].cat.categories

        density_values = np.zeros(adata.n_obs)

        for cat in categories:
            cat_mask = adata.obs[groupby] == cat
            embed_x = adata.obsm[f"{basis}"][cat_mask, components[0]]
            embed_y = adata.obsm[f"{basis}"][cat_mask, components[1]]

            dens_embed = kde(embed_x, embed_y)
            density_values[cat_mask] = dens_embed

        adata.obs[density_covariate] = density_values

    else:
        # calculate the density over the whole embedding without subsetting
        embed_x = adata.obsm[f"{basis}"][:, components[0]]
        embed_y = adata.obsm[f"{basis}"][:, components[1]]
        adata.obs[density_covariate] = kde(embed_x, embed_y)

    adata.uns[f"{density_covariate}"] = dict(
        covariate = groupby, components = components.tolist()
    )

    return 