import tensorflow_probability as tfp
from exo_search.entities.light_curve import iterate_lcs_from_dir, load_lcs_from_dir
from pathlib import Path
import gpflow
from exo_search.modeling.model import Model
from exo_search.modeling.manager import Manager

def fast_slow(lcs_path: Path, dst_path: Path):
    """Create models with with composite SE kernel.

    Args:
        lcs_path (Path): directory with light curves.
        dst_path (Path): destination for the models.
    """

    for lc in iterate_lcs_from_dir(lcs_path):
        kernel_fast = gpflow.kernels.SquaredExponential()
        kernel_slow = gpflow.kernels.SquaredExponential()

        # Fast kernel
        kernel_fast.lengthscales = gpflow.Parameter(
            2 / 24,
            transform=tfp.bijectors.SoftClip(
                gpflow.utilities.to_default_float(1 / 24),
                gpflow.utilities.to_default_float(5 / 24),
            ),
        )

        # Slow kernel
        kernel_slow.lengthscales = gpflow.Parameter(
            1,
            transform=tfp.bijectors.SoftClip(
                gpflow.utilities.to_default_float(0.5),
                gpflow.utilities.to_default_float(1.5),
            ),
        )

        model = Model(
            lc,
            lc.config_str + f",se_fast_slow_1h_5h_2h-0.5d_1.5d_1d",
            kernel=kernel_fast + kernel_slow,
        )
        model.save_to_file(
            dst_path / f"{model.config_str}.json",
            dst_path / "gpflow_models" / f"{model.config_str}.pickle",
        )

def custom(lc_path: Path, dst_path: Path):
    lcs = load_lcs_from_dir(lc_path)

    kernel = gpflow.kernels.SquaredExponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=10)
    manager = Manager()
    manager.create_models_from_lc(lcs, kernel=kernel, kernel_name="se")
    manager.save(dst_path)


# Example usage:

if __name__ == "__main__":
    fast_slow(Path("tmp/lcs/downsampled"), Path("tmp/models/fast_slow"))