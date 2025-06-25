import numpy as np


def z_score(data: np.ndarray) -> np.ndarray:
    """Z-score normalize the data. Mean is 0, std is 1.

    Args:
        data (np.ndarray): array.

    Returns:
        np.ndarray: normalized array.
    """

    return (data - np.nanmean(data)) / np.nanstd(data, ddof=1)
