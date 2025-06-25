from exo_search.entities.light_curve import LightCurve, load_lcs_from_dir
from exo_search.entities.exoplanet import Exoplanet
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List


def main(
    lc_path: Path,
    folded_lc_path: Path,
    plot_path: Path = None,
    disposition: str = None,
    file_format: str = "png",
    n: int = None,
) -> None:
    """Fold the light curve on transits.

    Args:
        lc_path (Path): directory with light curves.
        folded_lc_path (Path): directory for folded light curves.
        plot_path (Path, optional): directory for plots. Defaults to None.
        disposition (str, optional): fold only with light curves of this disposition. Defaults to None.
        file_format (str, optional): file format for the plots. Defaults to "png".
        n (int, optional): load only the n-th light curve. All light curves are loaded by default.
    """

    # Load light curves
    lcs = load_lcs_from_dir(lc_path, n)

    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)

    # Iterate over light curves
    for lc in lcs:
        # Iterate over exoplanets
        for e in lc.exoplanets:
            # Check the disposition
            if disposition is not None and e.disposition != disposition:
                continue

            # Check the exoplanet values
            if (
                e.transit_duration_h is None
                or e.period is None
                or e.transit_midpoint_BJD is None
            ):
                continue

            # Fold
            new_time, new_flux = fold(lc, e)
            if len(new_flux) == 0:
                continue

            # Combine
            time, flux = combine(new_time, new_flux)

            # Create new light curve
            split = lc.config_str.split(",")

            new_lc = LightCurve(
                lc.star_name,
                time,
                flux,
                lc.mission,
                lc.author,
                lc.cadence_period,
                lc.batch,
                [e],
                split[0] + f"_{e.name}," + ",".join(split[1:]),
            )
            # Save the light curve
            new_lc.save_to_file(folded_lc_path / f"{new_lc.config_str}.json")

            # Plot folded light curve
            if plot_path is not None:
                plot_folded(new_time, new_flux, lc, plot_path, file_format, e)


def combine(
    time: List[np.ndarray], flux: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine multiple time arrays and flux array into a single time and flux array.

    Args:
        time (List[np.ndarray]): list of time arrays.
        flux (List[np.ndarray]): list of flux arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Concatenated arrays.
    """
    time = np.concatenate(time)
    flux = np.concatenate(flux)
    sort_indexes = np.argsort(time)
    time = time[sort_indexes]
    flux = flux[sort_indexes]
    return (time, flux)


def fold(lc: LightCurve, e: Exoplanet) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fold the light curve by the exoplanet.

    Args:
        lc (LightCurve): light curve.
        e (Exoplanet): exoplanet.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: list of new time and flux arrays.
    """
    new_time = []
    new_flux = []
    padding = e.transit_duration_h / 24

    # Iterate over the transit positions
    transits = lc.get_transits(e)
    for transit in transits:
        # Create start and end time around the transit
        start = transit - padding
        end = transit + padding

        # Extract the time and flux values around the transit
        transit_time = []
        transit_flux = []
        for i, t in enumerate(lc.time):
            if start < t < end:
                transit_flux.append(lc.flux[i].item())
                transit_time.append(t.item())

        transit_flux = np.array(transit_flux)
        transit_time = np.array(transit_time)

        # Convert time (which is in days) range <-padding, padding>
        scaled_time = transit_time - transit

        if len(transit_flux) != 0:
            new_flux.append(transit_flux)
            new_time.append(scaled_time)

    return new_time, new_flux


def plot_folded(
    new_time: List[np.ndarray],
    new_flux: List[np.ndarray],
    new_lc: np.ndarray,
    plot_path: Path,
    file_format: str,
    e: Exoplanet,
) -> None:
    """PLot folded transit.

    Args:
        new_time (List[np.ndarray]): new time values per transit.
        new_flux (List[np.ndarray]): new flux values per transit.
        new_lc (np.ndarray): new light curve.
        plot_path (Path): directory for the figures.
        file_format (str): file format for the figure.
        e (Exoplanet): exoplanet.
    """
    # Create axis
    _, ax = plt.subplots()

    # Plot the time and flux values
    for i, flux in enumerate(new_flux):
        ax.scatter(
            new_time[i] * 24,
            flux,
            s=2,
            alpha=0.8,
            edgecolors="none",
        )

    # Labels
    ax.set_xlabel("Hours from transit midpoint")
    ax.set_ylabel("Flux")

    # Title and text
    ax.set_title(f"{new_lc.star_name} ({e.name})", pad=15)
    ax.text(
        0.5,
        1.02,
        new_lc.info,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=8,
    )

    plt.tight_layout()

    # Save figure
    plt.savefig(plot_path / f"{new_lc.config_str}.{file_format}")

    # Close figure
    plt.close()
