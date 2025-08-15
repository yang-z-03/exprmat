
from typing import Literal
from pathlib import Path
import exprmat.snapatac as internal


def export_coverage(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    bin_size: int = 10,
    blacklist: Path | None = None,
    normalization: Literal["RPKM", "CPM", "BPM"] | None = "RPKM",
    include_for_norm: list[str] | Path = None,
    exclude_for_norm: list[str] | Path = None,
    min_frag_length: int | None = None,
    max_frag_length: int | None = 2000,
    counting_strategy: Literal['fragment', 'insertion'] = 'fragment',
    smooth_base: int | None = None,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bigwig",
    output_format: Literal["bedgraph", "bigwig"] | None = None,
    compression: Literal["gzip", "zstandard"] | None = None,
    compression_level: int | None = None,
    tempdir: Path | None = None,
    n_jobs: int = 8,
) -> dict[str, str]:
    
    """
    Export and save coverage in a bedgraph or bigwig format file.

    This function first divides cells into groups based on the `groupby` parameter.
    It then independently generates the genome-wide coverage track (bigWig or bedGraph) for each group
    of cells. The coverage is calculated as the number of reads per bin, where bins are short 
    consecutive counting windows of a defined size. For paired-end data, the reads are extended to 
    the fragment length and the coverage is calculated as the number of fragments per bin.
    There are several options for normalization. The default is RPKM, which normalizes by the 
    total number of reads and the length of the region. The normalization can be disabled by 
    setting `normalization = None`.

    Parameters
    ----------
    groupby
        Group the cells. If a `str`, groups are obtained from `.obs[groupby]`.

    selections
        Export only the selected groups.

    bin_size
        Size of the bins, in bases, for the output of the bigwig/bedgraph file.

    blacklist
        A BED file containing the blacklisted regions.

    normalization
        Normalization method. If `None`, no normalization is performed. Options:
        - RPKM (per bin) = #reads per bin / (#mapped_reads (in millions) * bin length (kb)).
        - CPM (per bin) = #reads per bin / #mapped_reads (in millions).
        - BPM (per bin) = #reads per bin / sum of all reads per bin (in millions).

    include_for_norm
        A list of string (e.g., ["chr1:1-100", "chr2:2-200"]) or a BED file containing
        the genomic loci to include for normalization.
        If specified, only the reads that overlap with these loci will be used for normalization.
        A typical use case is to include only the promoter regions for the normalization.

    exclude_for_norm
        A list of string (e.g., ["chr1:1-100", "chr2:2-200"]) or a BED file containing
        the genomic loci to exclude for normalization.
        If specified, the reads that overlap with these loci will be excluded from normalization.
        If a read overlaps with both `include_for_norm` and `exclude_for_norm`, it will be excluded.

    min_frag_length
        Minimum fragment length to be included in the computation.

    max_frag_length
        Maximum fragment length to be included in the computation.

    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment" or "insertion". "fragment" means the feature counts are assigned based on the
        number of fragments that overlap with a region of interest. "insertion" means the feature 
        counts are assigned based on the number of insertions that overlap with a region of interest.

    smooth_base
        Length of the smoothing window in bases for the output of the bigwig/bedgraph file.

    out_dir
        Directory for saving the outputs.

    prefix
        Text added to the output file name.

    suffix
        Text added to the output file name.

    output_format
        Output format. If `None`, it is inferred from the suffix.

    compression
        Compression type. If `None`, it is inferred from the suffix.

    compression_level
        Compression level. 1-9 for gzip, 1-22 for zstandard.
        If `None`, it is set to 6 for gzip and 3 for zstandard.
        
    n_jobs
        Number of threads to use. If `<= 0`, use all available threads.

    Returns
    -------
    dict[str, str]
        A dictionary contains `(groupname, filename)` pairs. File names contains pure file
        names without directory roots.
    """

    if isinstance(groupby, str):
        groupby = adata.obs[groupby]
    if selections is not None:
        selections = set(selections)
    
    if output_format is None:
        output_format, inferred_compression = get_file_format(suffix)
        if output_format is None:
            raise ValueError("Output format cannot be inferred from suffix.")
        if compression is None:
            compression = inferred_compression

    n_jobs = None if n_jobs <= 0 else n_jobs
    return internal.export_coverage(
        adata, list(groupby), bin_size, out_dir, prefix, suffix, output_format, counting_strategy,
        selections, blacklist, normalization, include_for_norm, exclude_for_norm, min_frag_length,
        max_frag_length, smooth_base, compression, compression_level, tempdir, n_jobs,
    )



def get_file_format(suffix):
    suffix = suffix.lower()
    _suffix = suffix

    if suffix.endswith(".gz"):
        compression = "gzip"
        _suffix = suffix[:-3]

    elif suffix.endswith(".zst"):
        compression = "zstandard"
        _suffix = suffix[:-4]

    else: compression = None
    
    if suffix.endswith(".bw") or suffix.endswith(".bigwig"):
        format = "bigwig"
    elif _suffix.endswith(".bedgraph") or _suffix.endswith(".bg") or _suffix.endswith(".bdg"):
        format = "bedgraph"
    else: format = None
    
    return format, compression