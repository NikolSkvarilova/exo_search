import numpy as np
import matplotlib.pyplot as plt
from exo_search.entities.light_curve import iterate_lcs_from_dir, LightCurve
from pathlib import Path
import pandas as pd
from typing import List, Tuple


def main(
    lc_path: Path,
    plot_path: Path = None,
    file_format: str = "png",
    report_filepath: Path = None,
    results_filepath: Path = None,
) -> None:
    """Calculate the fft of light curves.

    Args:
        lc_path (Path): directory with light curves.
        plot_path (Path, optional): directory for figures. Defaults to None.
        file_format (str, optional): file format for figures. Defaults to "png".
        report_filepath (Path, optional): file name for the plot with found periods. Default to None.
        results_filepath (Path, optional): csv with found periods.
    """
    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)

    predicted_periods = []

    exoplanet_periods = []
    predicted_periods_for_exoplanets = []
    colors = []

    for lc in iterate_lcs_from_dir(lc_path):
        predicted_periods.append(
            [
                lc.star_name,
                lc.mission,
                lc.author,
                lc.cadence_period,
                "+".join(map(str, lc.batch)),
            ]
        )
        # Calculate FFT
        Fs = 1 / lc.cadence_period
        lc.time, lc.flux = lc.fill_gaps()
        lc.flux = np.nan_to_num(lc.flux, nan=0.0)
        f, X = fft(lc.flux, Fs)
        f = f[1:]
        X = X[1:]

        # Covert frequencies to days
        f_days = (1 / f) / (60 * 60 * 24)

        # Plot FFT
        if plot_path is not None:
            plot_fft(f_days, X, lc, plot_path, file_format, "Days")
            plot_fft(f, X, lc, plot_path, file_format, "Hz")

        # Compute found period
        if report_filepath is not None:
            # Find the highest peak and corresponding period
            best_index = np.argmax(X)
            found_period = f_days[best_index]
            predicted_periods[-1].append(found_period)
            for e in lc.exoplanets:
                if e.period is None or e.period > 100:
                    continue

                exoplanet_periods.append(e.period)
                predicted_periods_for_exoplanets.append(found_period)

                if len(lc.exoplanets) > 1:
                    colors.append("red")
                else:
                    colors.append("blue")

    # Plot found periods
    if report_filepath is not None:
        report_filepath.parent.mkdir(parents=True, exist_ok=True)
        plot_found_periods(
            exoplanet_periods,
            predicted_periods_for_exoplanets,
            colors,
            report_filepath,
        )

    # Save found periods
    if results_filepath is not None:
        results_filepath.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(predicted_periods)
        df.columns = [
            "star_name",
            "mission",
            "author",
            "cadence_period",
            "batch",
            "fft_found_period",
        ]
        df.to_csv(results_filepath, index=False)


def plot_fft(
    f: np.ndarray,
    X: np.ndarray,
    lc: LightCurve,
    plot_path: Path,
    file_format: str,
    xlabel: str,
) -> None:
    """Plot the fft spectrum.

    Args:
        f (np.ndarray): frequencies.
        X (np.ndarray): magnitudes.
        lc (LightCurve): light curve.
        plot_path (Path): directory for the figure.
        file_format (str): file format for the figure.
        xlabel (str): x label for the figure.
    """
    # Create axis
    _, ax = plt.subplots()

    # Plot FFT
    marker, stem, base = ax.stem(f, X, "blue", basefmt=" ")
    plt.setp(marker, markersize=2)
    plt.setp(stem, linewidth=0.5)

    # Set title and text
    ax.set_title("FFT Spectrum", pad=15)
    ax.text(
        0.5,
        1.02,
        lc.info,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=8,
    )

    # Set labels
    ax.set_ylabel("X")
    ax.set_xlabel(xlabel)

    plt.tight_layout()

    # Save figure
    plt.savefig(plot_path / f"{lc.config_str}_FFT_{xlabel}.{file_format}")

    # Close figure
    plt.close()


def plot_found_periods(
    correct_periods: List[float],
    predicted_periods: List[float],
    colors: List[str],
    report_filepath: Path,
) -> None:
    """Plot the found periods against real periods.

    Args:
        correct_periods (List[float]): list of correct periods.
        predicted_periods (List[float]): list of found periods.
        colors (List[str]): color for each period.
        report_filepath (Path): filename for the figure.
    """
    # Create the axis
    _, ax = plt.subplots()

    # Plot the periods
    for i, period in enumerate(correct_periods):
        ax.scatter(
            period,
            predicted_periods[i],
            s=4,
            color=colors[i],
            edgecolors="none",
            alpha=0.7,
        )

    # Set labels
    ax.set_xlabel("Exoplanet Period [Days]")
    ax.set_ylabel("Found Period [Days]")

    # Plot harmonics
    start = 0
    end = 15
    n = 6
    for i in range(n):
        ax.plot(
            [start, end], [start, end / (i + 1)], color="blue", linewidth=0.5, alpha=0.3
        )

    # Limit the axis (some periods can be large)
    ax.set_xlim(left=0, right=15)
    ax.set_ylim(bottom=0, top=7.5)

    # Set title
    ax.set_title(f"Periods found using FFT")

    # Save figure
    plt.savefig(report_filepath)

    # Close figure
    plt.close()


def fft(x: np.ndarray, Fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute single-sided fft.

    Args:
        x (np.ndarray): signal sampels.
        Fs (float): sampling frequency in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]: frequencies [Hz] and fft amplitudes.
    """
    # Number of samples
    N = len(x)
    # FFT spectrum
    X = np.fft.fft(x)
    # Positive frequency values
    f = Fs * np.arange(0, N // 2 + 1) / N
    # Normalized magnitude spectrum
    X_abs_norm = np.abs(X / N)
    # Use only half
    X_half = X_abs_norm[: N // 2 + 1]
    # Double the amplitudes
    X_half[1:-1] = 2 * X_half[1:-1]
    return f, X_half
