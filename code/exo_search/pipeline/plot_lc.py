from exo_search.entities.light_curve import load_lcs_from_dir
from pathlib import Path


def main(
    lc_path: Path,
    plot_path: Path,
    file_format: str = "png",
    n: int = None,
    plot_exoplanets: bool = True,
) -> None:
    """Plot light curves.

    Args:
        lc_path (Path): directory with light curves.
        plot_path (Path): directory for the plots.
        file_format (str, optional): file format for the plots. Defaults to "png".
        n (int): load only the n-th light curve. All light curves from the directory are loaded by default.
        plot_exoplanets (bool): add transit lines into the plot.
    """

    lcs = load_lcs_from_dir(lc_path, n)
    for lc in lcs:
        lc.plot(
            save_fig=plot_path / f"{lc.config_str}.{file_format}",
            show_fig=False,
            plot_exoplanets=plot_exoplanets,
        )
