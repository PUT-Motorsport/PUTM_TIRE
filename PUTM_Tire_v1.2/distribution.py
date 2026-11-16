"""Histogram preparation for measurement channels, independent of the UI."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Histogram:
    edges: np.ndarray
    counts: np.ndarray
    total: int
    finite: int
    below: int = 0
    above: int = 0


def make_histogram(values, *, focus=False, max_bins=48) -> Histogram:
    """Count finite samples with bounded, data-dependent bin widths.

    Focus uses the 1st–99th percentile for the visible bins and reports both
    tails separately, so samples outside the view are never silently lost.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    total = int(values.size)
    finite = int(finite_values.size)
    if not finite:
        return Histogram(np.array([]), np.array([], dtype=int), total, 0)

    distinct = np.unique(finite_values)
    low, high = float(distinct[0]), float(distinct[-1])
    below = above = 0
    visible = finite_values

    if focus and distinct.size > 12:
        q_low, q_high = np.percentile(finite_values, [1, 99])
        if np.isfinite(q_low) and np.isfinite(q_high) and q_low < q_high:
            low, high = float(q_low), float(q_high)
            below = int(np.count_nonzero(finite_values < low))
            above = int(np.count_nonzero(finite_values > high))
            visible = finite_values[(finite_values >= low) & (finite_values <= high)]
            distinct = np.unique(visible)

    if low == high:
        padding = max(abs(low) * 0.01, 0.5)
        edges = np.array([low - padding, high + padding])
    elif distinct.size <= 12 and not focus:
        midpoints = (distinct[:-1] + distinct[1:]) / 2
        edges = np.concatenate(([distinct[0] - (midpoints[0] - distinct[0])], midpoints,
                                [distinct[-1] + (distinct[-1] - midpoints[-1])]))
    else:
        q25, q75 = np.percentile(visible, [25, 75])
        width = 2 * (q75 - q25) / np.cbrt(visible.size)
        if width > 0 and np.isfinite(width):
            n_bins = int(np.ceil((high - low) / width))
        else:
            n_bins = int(np.ceil(np.log2(visible.size) + 1))
        n_bins = int(np.clip(n_bins, 8, max_bins))
        edges = np.linspace(low, high, n_bins + 1)

    counts, edges = np.histogram(visible, bins=edges)
    return Histogram(edges, counts, total, finite, below, above)
