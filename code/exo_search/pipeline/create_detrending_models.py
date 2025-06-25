from exo_search.modeling.manager import Manager
from exo_search.entities.light_curve import load_lcs_from_dir
from pathlib import Path
from exo_search.utils.kernels import get_kernel
from typing import Tuple
import gpflow
import tensorflow_probability as tfp


def create_se_kernel(min: float, max: float, default: float) -> gpflow.kernels.SquaredExponential:
    """Create SE kernel with lengthscale as a range with default value.

    Args:
        min (float): min value in the range.
        max (float): max value in the range.
        default (float): default value.

    Returns:
        gpflow.kernels.SquaredExponential: SE kernel.
    """

    kernel = gpflow.kernels.SquaredExponential()
    kernel.lengthscales = gpflow.Parameter(
        default,
        transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(min),
            gpflow.utilities.to_default_float(max),
        ),
    )

    return kernel


def main(
    lc_path: Path,
    model_path: Path,
    fast: Tuple[float, float, float],
    slow: Tuple[float, float, float],
    kernel_name: str = None,
    n: int = None,
) -> None:
    """Create detrending models.

    Args:
        lc_path (Path): directory with light curves.
        model_path (Path): directory for the newly created models.
        fast (Tuple[float, float, float]): min, max, and default lengthscale for the fast kernel.
        slow (Tuple[float, float, float]): min, max, and default lengthscale for the slow kernel.
        kernel_name (str, optional): kernel name. Defaults to se_fast_slow_<default_fast>_<min_fast>_<max_fast>-<default_slow>_<min_slow>_<max_slow>.
        n (int, optional): load only n-th light curve from the directory (creates only 1 model). All light curves are loaded by default.
    """

    # Load light curves
    lcs = load_lcs_from_dir(lc_path, n)
    # Create models
    manager = Manager()

    # Create composite kernel
    kernel = create_se_kernel(*fast) + create_se_kernel(*slow)
    if kernel_name is None:
        kernel_name = f"se_fast_slow_{round(fast[0], 2)}d_{round(fast[1], 2)}d_{round(fast[2], 2)}d-_{round(slow[0], 2)}d_{round(slow[1], 2)}d_{round(slow[2], 2)}d"

    # Create models
    manager.create_models_from_lc(
        lcs,
        kernel,
        kernel_name,
        model_path=model_path,
    )
    # Save models
    manager.save(model_path)
