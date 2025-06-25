from exo_search.entities.light_curve import iterate_lcs_from_dir
from pathlib import Path
from typing import Callable, Dict, Any


def main(
    lc_path: Path, dst_path: Path, method: Callable, params: Dict[str, Any]
) -> None:
    """Downsample light curves. Adds the downsampling information to the config string.

    Args:
        lc_path (Path): directory with light curves.
        dst_path (Path): directory for downsampled light curves.
        method (Callable): method for downsampling.
        params (Dict[str: Any]): parameters for the downsampling function.
    """
    dst_path.mkdir(parents=True, exist_ok=True)

    for lc in iterate_lcs_from_dir(lc_path):
        # Downsample
        lc_downsampled = lc.downsample(method, **params)

        # Add info into the config string
        lc_downsampled.config_str += f",{method.__name__},{';'.join([f'{name}={value}' for name, value in params.items()])}"
        
        # Save to file
        lc_downsampled.save_to_file(dst_path / f"{lc_downsampled.config_str}.json")
