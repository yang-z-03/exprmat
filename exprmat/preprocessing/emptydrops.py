
from __future__ import annotations

import numpy as np
import scipy.stats as sp_stats
from exprmat.ansi import warning, error, info


MIN_RECOVERED_CELLS_PER_GEM_GROUP = 50
MAX_RECOVERED_CELLS_PER_GEM_GROUP = 1 << 18


class metrics:
    def update(self, other):
        for k, v in other.__dict__.items():
            if v is not None:
                setattr(self, k, v)


class barcode_filter(metrics):
    def __init__(self, default_value: int = 0):
        self.filtered_bcs = default_value
        self.filtered_bcs_lb = default_value
        self.filtered_bcs_ub = default_value
        self.filtered_bcs_var = default_value
        self.filtered_bcs_cv = float(default_value)
        self.filtered_bcs_cutoff = default_value

    @staticmethod
    def init_with_constant_call(n_bcs: int):
        res = barcode_filter()
        res.filtered_bcs = n_bcs
        res.filtered_bcs_lb = n_bcs
        res.filtered_bcs_ub = n_bcs
        res.filtered_bcs_var = 0
        res.filtered_bcs_cv = 0
        res.filtered_bcs_cutoff = 0
        return res


def filter_cellular_barcodes_ordmag(
    bc_counts: np.ndarray[int, np.dtype[np.int_]],
    recovered_cells: int | None,
    chemistry_description: str | None = None,
    num_probe_barcodes: int | None = None,
    ordmag_recovered_cells_quantile: float = .99,
    num_bootstrap_samples: int = 100,
):
    """
    All barcodes that are close to within an order of magnitude of a top barcode.
    Takes all barcodes that are close to within an order of magnitude of a
    top barcode that likely represents a cell.
    """

    rs = np.random.RandomState(0)
    metrics = barcode_filter(0)

    nonzero_bc_counts = bc_counts[bc_counts > 0]
    if len(nonzero_bc_counts) == 0:
        warning("All barcodes do not have enough reads for ordmag, allowing no bcs through")
        return [], metrics

    if recovered_cells is None:
        # Set the most cells to examine based on the empty drops range for this chemistry
        max_expected_cells = (
            min(
                get_empty_drops_range(chemistry_description, num_probe_barcodes)[0],
                MAX_RECOVERED_CELLS_PER_GEM_GROUP,
            )
            if (chemistry_description is not None)
            else MAX_RECOVERED_CELLS_PER_GEM_GROUP
        )

        recovered_cells, loss = np.mean(
            np.array([
                estimate_recovered_cells_ordmag(
                    rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), max_expected_cells,
                    ordmag_recovered_cells_quantile,
                ) for _ in range(num_bootstrap_samples)
            ]), axis = 0,
        )

        recovered_cells = max(int(np.round(recovered_cells)), MIN_RECOVERED_CELLS_PER_GEM_GROUP)
        info(f"found recovered cells = {recovered_cells} with loss = {loss}")
    
    else:
        recovered_cells = max(recovered_cells, MIN_RECOVERED_CELLS_PER_GEM_GROUP)
        info(f"using provided recovered cells = {recovered_cells}")

    baseline_bc_idx = int(np.round(float(recovered_cells) * (1 - ordmag_recovered_cells_quantile)))
    baseline_bc_idx = min(baseline_bc_idx, len(nonzero_bc_counts) - 1)

    # Bootstrap sampling; run algo with many random samples of the data
    top_n_boot = np.array([
        find_within_ordmag(
            rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), baseline_bc_idx
        ) for _ in range(num_bootstrap_samples)
    ] )

    metrics.update(summarize_bootstrapped_top_n(top_n_boot, nonzero_bc_counts))

    # Get the filtered barcodes
    top_n = metrics.filtered_bcs
    top_bc_idx = np.sort(np.argsort(bc_counts, kind = "stable")[::-1][0:top_n])
    assert top_n <= len(nonzero_bc_counts), "invalid selection of 0-count barcodes!"
    return top_bc_idx, metrics


def get_empty_drops_range(chemistry_description: str, num_probe_bcs: int | None) -> tuple[int, int]:
    """
    Gets the range of values to use for empty drops background given a chemistry description.
    Cell Ranger's grid-search ranges from 2 to ~45,000 cells for NextGEM chemistries and 
    2 to ~80,000 for GEM-X chemistries.

    chemistry_description: 
        A string describing the chemistry
    """

    if chemistry_description == "v3LT":
        N_PARTITIONS = 9000
    elif chemistry_description == "v4":
        N_PARTITIONS = 80000 * num_probe_bcs if num_probe_bcs and num_probe_bcs > 1 else 160000
    else: N_PARTITIONS = 45000 * num_probe_bcs if num_probe_bcs and num_probe_bcs > 1 else 90000
    return (N_PARTITIONS // 2, N_PARTITIONS)


def estimate_recovered_cells_ordmag(
    nonzero_bc_counts,
    max_expected_cells: int,
    ordmag_recovered_cells_quantile: float,
):
    """ Estimate the number of recovered cells by trying to find ordmag(recovered) =~ filtered. """
    # - search for a result such that some loss(recovered_cells, filtered_cells) is minimized.
    # - here i'm using (obs - exp)**2 / exp, which approximates a proportion for small differences
    #   but blows up for large differences.
    # - test over a log2-spaced range of values from 1..262_144
    recovered_cells = np.linspace(1, np.log2(max_expected_cells), 2000)
    recovered_cells = np.unique(np.round(np.power(2, recovered_cells)).astype(int))
    baseline_bc_idx = np.round(recovered_cells * (1 - ordmag_recovered_cells_quantile))
    baseline_bc_idx = np.minimum(baseline_bc_idx.astype(int), len(nonzero_bc_counts) - 1)
    filtered_cells = find_within_ordmag(nonzero_bc_counts, baseline_bc_idx)
    loss = np.power(filtered_cells - recovered_cells, 2) / recovered_cells
    idx = np.argmin(loss)
    return recovered_cells[idx], loss[idx]


def find_within_ordmag(x, baseline_idx):
    x_ascending = np.sort(x)
    # add +1 as we're getting from the other side
    baseline = x_ascending[-(baseline_idx + 1)]
    cutoff = np.maximum(1, np.round(0.1 * baseline)).astype(int)
    # return the index corresponding to the cutoff in descending order
    return len(x) - np.searchsorted(x_ascending, cutoff)


def summarize_bootstrapped_top_n(top_n_boot, nonzero_counts):
    top_n_bcs_mean = np.mean(top_n_boot)
    top_n_bcs_var = np.var(top_n_boot)
    top_n_bcs_sd = np.sqrt(top_n_bcs_var)
    result = barcode_filter()
    result.filtered_bcs_var = top_n_bcs_var
    result.filtered_bcs_cv = top_n_bcs_sd / top_n_bcs_mean
    result.filtered_bcs_lb = np.round(sp_stats.norm.ppf(0.025, top_n_bcs_mean, top_n_bcs_sd), 0)
    result.filtered_bcs_ub = np.round(sp_stats.norm.ppf(0.975, top_n_bcs_mean, top_n_bcs_sd), 0)

    nbcs = int(np.round(top_n_bcs_mean))
    result.filtered_bcs = nbcs

    # make sure that if a barcode with count x is selected, we select all barcodes with count >= x
    # this is true for each bootstrap sample, but is not true when we take the mean

    if nbcs > 0:
        order = np.argsort(nonzero_counts, kind="stable")[::-1]
        sorted_counts = nonzero_counts[order]

        cutoff = sorted_counts[nbcs - 1]
        index = nbcs - 1
        while (index + 1) < len(sorted_counts) and sorted_counts[index] == cutoff:
            index += 1
            # if we end up grabbing too many barcodes, revert to initial estimate
            if (index + 1 - nbcs) > 0.20 * nbcs:
                return result
            result.filtered_bcs = index + 1
            result.filtered_bcs_cutoff = cutoff
            
    return result

