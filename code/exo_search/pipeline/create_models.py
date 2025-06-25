from exo_search.modeling.manager import Manager
from exo_search.entities.light_curve import load_lcs_from_dir
from pathlib import Path
from exo_search.utils.kernels import get_kernel


def main(lc_path: Path, model_path: Path, kernel_name: str, n: int = None) -> None:
    """Create models.

    Args:
        lc_path (Path): directory with light curves.
        model_path (Path): directory for the newly created models.
        kernel_name (str): name of the kernel, from the standard dictionary.
        n (int, optional): load only n-th light curve from the directory (creates only 1 model). All light curves are loaded by default.
    """
    # Load light curves
    lcs = load_lcs_from_dir(lc_path, n)
    # Create models
    manager = Manager()
    kernel = get_kernel(kernel_name)
    manager.create_models_from_lc(
        lcs,
        kernel,
        kernel_name,
        model_path=model_path,
    )
    # Save models
    manager.save(model_path)
