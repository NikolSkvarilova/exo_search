import math


def compare_floats(f1: float, f2: float) -> bool:
    """Comparing floats using math.isclose. Checks for None.

    Args:
        f1 (float): first float.
        f2 (float): second float.

    Returns:
        bool: True if equal/close.
    """
    if f1 is None or f2 is None:
        return f1 == f2

    return math.isclose(f1, f2)
