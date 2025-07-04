
from typing import Literal
from pathlib import Path
import numpy as np
from anndata import AnnData

import exprmat.snapatac as internal
from exprmat.utils import get_file_format, anndata_rs_par
from exprmat.data.finders import get_genome_gff_fname


def fragment_size_distribution(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    max_recorded_size: int = 1000,
    add_key: str = "frag.sizes",
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """ 
    Compute the fragment size distribution of the dataset. 

    This function computes the fragment size distribution of the dataset.
    Note that it does not operate at the single-cell level. The result is stored in a vector 
    where each element represents the number of fragments and the index represents the 
    fragment length. The first posision of the vector is reserved for fragments with size 
    larger than the `max_recorded_size` parameter.

    The typical fragment size distribution in ATAC-seq data exhibits several key characteristics:

    - Nucleosome-Free Region (NFR): The majority of fragments are short, typically around 
      80-300 base pairs (bp) in length. These short fragments represent regions of open 
      chromatin where DNA is relatively accessible and not bound to nucleosomes. These regions 
      are often referred to as nucleosome-free regions (NFRs) and correspond to regions of 
      active transcriptional regulation.

    - Mono-Nucleosome Peaks: There is often a peak in the fragment size distribution at 
      around 147-200 bp, which corresponds to the size of a single nucleosome wrapped DNA. 
      These fragments result from the cutting of DNA by the transposase enzyme when it 
      encounters a nucleosome, causing the DNA to be protected and resulting in fragments 
      of the same size.

    - Di-Nucleosome Peaks: In addition to the mono-nucleosome peak, you may also observe a 
      smaller peak at approximately 300-400 bp, corresponding to di-nucleosome fragments. 
      These fragments occur when transposase cuts the DNA between two neighboring nucleosomes.

    - Large Fragments: Some larger fragments, greater than 500 bp in size, may be observed. 
      These fragments can result from various sources, such as the presence of longer 
      stretches of open chromatin or technical artifacts.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` * `n_vars`.
        Rows correspond to cells and columns to regions. `adata` could also be a list of 
        AnnData objects. In this case, the function will be applied to each AnnData object 
        in parallel.

    max_recorded_size
        The maximum fragment size to record in the result. Fragments with length larger 
        than `max_recorded_size` will be recorded in the first position of the result vector.

    Returns
    -------
    np.ndarray | list[np.ndarray] | None
    """

    if isinstance(adata, list):
        return anndata_rs_par(
            adata, lambda x: fragment_size_distribution(
                x, add_key = add_key, 
                max_recorded_size = max_recorded_size, 
                inplace = inplace
            ), n_jobs = n_jobs,
        )
    
    else:
        result = np.array(internal.fragment_size_distribution(adata, max_recorded_size))
        if inplace: adata.uns[add_key] = result
        else: return result


def tss_enrichment(
    adata: internal.AnnData | list[internal.AnnData],
    assembly,
    *,
    exclude_chroms: list[str] | str | None = ["chrM", "M"],
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """ 
    Compute the TSS enrichment score (TSSe) for each cell.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float]] | list[tuple[np.ndarray, tuple[float, float]]] | None
        If `inplace = True`, cell-level tss enrichment scores are computed and stored in 
        `adata.obs['tsse']`. library-level tsse scores are stored in `adata.uns['tsse']`.
        Fraction of fragments overlapping TSS are stored in `adata.uns['pct.tss.overlap']`.
    """

    gene_anno = get_genome_gff_fname(assembly)
 
    if isinstance(adata, list):
        result = anndata_rs_par(
            adata,
            lambda x: tss_enrichment(x, gene_anno, exclude_chroms = exclude_chroms, inplace = inplace),
            n_jobs = n_jobs,
        )
    
    else:
        result = internal.tss_enrichment(adata, gene_anno, exclude_chroms)
        result['tsse'] = np.array(result['tsse'])
        result['tss.profile'] = np.array(result['TSS_profile'])
        if inplace:
            adata.obs["tsse"] = result['tsse']
            adata.uns['tsse'] = result['library_tsse']
            adata.uns['pct.tss.overlap'] = result['frac_overlap_TSS']
            adata.uns['tss.profile'] = result['TSS_profile']
    
    if inplace: return None
    else: return result


def frip(
    adata: internal.AnnData | list[internal.AnnData],
    regions: dict[str, Path | list[str]],
    *,
    normalized: bool = True,
    count_as_insertion: bool = False,
    inplace: bool = True,
    n_jobs: int = 8,
) -> dict[str, list[float]] | list[dict[str, list[float]]] | None:
    """ 
    Add fraction of reads in peaks (FRiP) to the AnnData object.
    Since we cannot call peaks before clustering, consensus peaks may be influenced more
    by the cells with higher proportions. This is not what we want for QC and filtering.
    This implementation uses fraction of reads in predefined promoter regions instead
    of classical FRiP definition. It is still doubtful whether this metric yield rational
    results. See discussion here <https://www.archrproject.com/bookdown/per-cell-quality-control.html>

    Parameters
    ----------
    regions
        A dictionary containing the peak sets to compute FRiP.
        The keys are peak set names and the values are either a bed file name or a list of
        strings representing genomic regions. For example,
        `{ "pct.promoter": "promoter.bed", "pct.enhancer": ["chr1:100-200", "chr2:300-400"] }`.

    normalized
        Whether to normalize the counts by the total number of fragments.
        If False, the raw number of fragments in peaks will be returned.

    count_as_insertion
        Whether to count transposition events instead of fragments. Transposition
        events are located at both ends of fragments.

    Returns
    -------
    dict[str, list[float]] | list[dict[str, list[float]]] | None
        If `inplace = True`, directly adds the results to `adata.obs`.
        Otherwise return a dictionary containing the results.
    """

    for k in regions.keys():
        if isinstance(regions[k], str) or isinstance(regions[k], Path):
            regions[k] = internal.read_regions(Path(regions[k]))
        elif not isinstance(regions[k], list):
            regions[k] = list(iter(regions[k]))

    if isinstance(adata, list):
        result = anndata_rs_par(
            adata, lambda x: frip(x, regions, inplace = inplace),
            n_jobs = n_jobs,
        )

    else:
        result = internal.add_frip(adata, regions, normalized, count_as_insertion)
        if inplace:
            for k, v in result.items():
                adata.obs[k] = v
    
    if inplace: return None
    else: return result


def summary_chromosomes(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    mode: Literal['sum', 'mean', 'count'] = 'count',
    n_jobs: int = 8,
) -> dict[str, np.ndarray]:
    """ 
    Compute the cell level summary statistics by chromosome.

    Parameters
    ----------
    mode
        The summary statistics to compute. It can be one of the following:
        - 'sum': Sum of the values.
        - 'mean': Mean of the values.
        - 'count': Count of the values.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the summary statistics for each chromosome.
        The keys are chromosome names and the values are the summary statistics.
    """

    if isinstance(adata, list):
        return anndata_rs_par(
            adata, lambda x: summary_chromosomes(x, mode = mode),
            n_jobs = n_jobs,
        )
    
    else: return internal.summary_by_chrom(adata, mode)
