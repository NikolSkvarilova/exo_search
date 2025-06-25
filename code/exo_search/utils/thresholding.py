import numpy as np


def moving_avg_gap(x: np.array, gap: int, ref: int, bias: float) -> np.array:
    """Calculate threshold using moving average with gap.

    Args:
        x (np.array): array to calculate the threshold over.
        gap (int): number of gap cells.
        ref (int): number of reference cells.
        bias (float): bias.

    Returns:
        np.array: threshold.
    """
    t = []

    for i in range(gap + ref, len(x) - (gap + ref)):
        left = x[i - gap - ref : i - gap]
        right = x[i + gap + 1 : i + gap + ref + 1]
        s = np.sum(left) + np.sum(right)
        t.append(bias * s / (2 * ref))

    return t
