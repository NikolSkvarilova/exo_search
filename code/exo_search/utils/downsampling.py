from typing import Tuple, Callable
import numpy as np


def every_nth_1D(x: np.ndarray, step: int = 10) -> np.ndarray:
    """Every n-th.

    Args:
        x (np.ndarray): array.
        step (int, optional): the n. Defaults to 10.

    Returns:
        np.ndarray: new array.
    """
    return x[:: int(step)]


def every_nth(
    x: np.ndarray, y: np.ndarray, step: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Every n-th over x and y values.

    Args:
        x (np.ndarray): array.
        y (np.ndarray): array.
        step (int, optional): the n. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: new arrays.
    """
    return (every_nth_1D(x, step), every_nth_1D(y, step))


def _moving_window(
    function_x: Callable,
    function_y: Callable,
    x: np.array,
    y: np.array,
    window: int,
    step: int,
    gap_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Moving window over 2D data. If gap between x values is larger than gap_threshold,
    gap is detected and instead of jumping by step in x, jump at the beginning of the new section
    of data is performed.

    Args:
        function_x (Callable): how to handle x-values in the window.
        function_y (Callable): how to handle y-values in the window.
        x (np.array): array.
        y (np.array): array.
        window (int): size of window.
        step (int): the n.
        gap_threshold (float): size of gap in the x values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: new arrays.
    """
    new_x = []
    new_y = []

    window = int(window)
    step = int(step)

    # Position in the list
    index = 0
    last_windowed_index = None

    while index < len(x):
        buffer_x = []
        buffer_y = []
        gap = False
        for i in range(index, index + window):
            if i >= len(x) or (
                len(buffer_x) > 0 and x[i] - buffer_x[-1] > gap_threshold
            ):
                gap = True
                break

            buffer_x.append(x[i])
            buffer_y.append(y[i])
            last_windowed_index = i

        if len(buffer_x) == 0:
            break

        new_x.append(function_x(buffer_x))
        new_y.append(function_y(buffer_y))

        # Move in the array
        # If gap was reached, go at the start of the new segment
        # If no gap, move by step
        if gap:
            index += len(buffer_x)
        else:
            index += step

        if last_windowed_index == len(x) - 1:
            break

    return (np.array(new_x), np.array(new_y))


def moving_avg(
    x: np.array, y: np.array, window: int, step: int, gap_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Moving window over 2D data. If gap between x values is larger than gap_threshold,
    gap is detected and instead of jumping by step in x, jump at the beginning of the new section
    of data is performed.

    Args:
        x (np.array): array.
        y (np.array): array.
        window (int): size of window.
        step (int): the n.
        gap_threshold (float): size of gap in the x values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: new arrays.
    """
    return _moving_window(np.mean, np.mean, x, y, window, step, gap_threshold)
