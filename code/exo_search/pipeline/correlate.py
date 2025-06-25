from exo_search.entities.light_curve import load_lcs_from_dir
from exo_search.modeling.manager import Manager
from pathlib import Path

import numpy as np
from exo_search.utils.thresholding import moving_avg_gap
from copy import deepcopy
import pandas as pd
from typing import List, Tuple, Any, Dict
import json
import threading
import matplotlib.pyplot as plt


def main(
    lc_path: Path,
    transit_model_path: Path,
    bias: float,
    gap: int = None,
    ref: int = None,
    plot_path: Path = None,
    results_path: Path = None,
    prediction_path: Path = None,
    file_format: str = "png",
    compare_existing: bool = False,
    n: int = None,
    n_threads: int = 0,
) -> None:
    """Correlate light curves with transit models.

    Args:
        lc_path (Path): directory with the light curves.
        transit_model_path (Path): directory with the transit models.
        bias (float): numeric value used for the bias. Determines the height of the threshold.
        gap (int, optional): number of gap cells. 1.5 times the width of the transit by default.
        ref (int, optional): number of reference cells. 6 times the width of the transit by default.
        plot_path (Path, optional): directory for the figures. Defaults to None.
        results_path (Path, optional): directory for the csv results. Defaults to None.
        prediction_path (Path, optional): directory with the predictions. Defaults to <transit_model_path>/predictions.
        file_format (str, optional): file format for the plots. Defaults to "png".
        n (int, optional): load only the n-th light curve from the directory. All light curves are loaded by default.
        n_threads (int, optional): run the calculation in threads. Defaults to 0.
    """

    # Matplotlib throws error when plotting and multiple threads are running
    if n_threads > 0 and plot_path is not None:
        raise RuntimeError("ERROR: unable to use threads and produce plots at the same time.")

    if prediction_path is None:
        prediction_path = transit_model_path / "predictions"

    if results_path is not None:
        (results_path / "corr_results").mkdir(parents=True, exist_ok=True)

    # Load light curves
    lcs = load_lcs_from_dir(lc_path, n)
    if len(lcs) == 0:
        print("Loaded 0 light curves!")
        return

    # Load models and predictions
    m = Manager()
    m.load(transit_model_path)
    if len(m.models) == 0:
        print("Loaded 0 transit models!")
        return
    m.load_predictions(prediction_path)

    if n_threads > 0:
        # Run in threads
        lcs_count = len(lcs)
        n_threads = min(n_threads, lcs_count)
        lcs_count_per_thread = lcs_count // n_threads
        threads = []
        for i in range(n_threads):
            lcs_for_threads = lcs[
                i * lcs_count_per_thread : (i + 1) * lcs_count_per_thread
            ]
            # If it is the last thread, do the rest of the models
            if i == n_threads - 1:
                lcs_for_threads = lcs[i * lcs_count_per_thread :]

            threads.append(
                threading.Thread(
                    target=correlate,
                    args=(
                        lcs_for_threads,
                        m.models,
                        plot_path,
                        compare_existing,
                        results_path,
                        file_format,
                        gap,
                        ref,
                        bias,
                    ),
                )
            )

        # Start threads
        for i, thread in enumerate(threads):
            print(f"Starting thread {i + 1} / {n_threads}")
            thread.start()

        # Join threads
        for thread in threads:
            thread.join()
    else:
        # Calculate without threads
        correlate(
            lcs,
            m.models,
            plot_path,
            compare_existing,
            results_path,
            file_format,
            gap,
            ref,
            bias,
        )


def correlate(
    lcs: List["LightCurve"],
    transit_models: List["Model"],
    plot_path: Path,
    compare_existing: bool,
    results_path: Path,
    file_format: str,
    gap: int,
    ref: int,
    bias: float,
) -> None:
    """Perform correlation.

    Args:
        lcs (List[LightCurve]): light curves.
        transit_models (List[Model]): transit models.
        plot_path (Path): directory for plots.
        compare_existing (bool): compare found transits with known transits.
        results_path (Path): directory for results.
        file_format (str): file format for plots.
        gap (int): gap for moving average.
        ref (int): ref for moving average.
        bias (float): bias for moving average.
    """

    # Iterate over each light curve
    for original_lc in lcs:
        print(f"Processing lc {original_lc.info}")

        # Initialize list/dictionaries with results
        corr_results = {
            "star_name": original_lc.star_name,
            "mission": original_lc.mission,
            "author": original_lc.author,
            "cadence_period": original_lc.cadence_period,
            "batch": "+".join(map(str, original_lc.batch)),
            "t_positions": {},
            "corr_threshold": {},
        }
        ok_detected = [
            original_lc.star_name,
            original_lc.mission,
            original_lc.author,
            original_lc.cadence_period,
            "+".join(map(str, original_lc.batch)),
        ]
        detectable = ok_detected.copy()
        detected = ok_detected.copy()

        # Iterate over transit models
        transit_index = 0
        for transit in transit_models:
            print(f"Processing transit {transit_index + 1} / {len(transit_models)}")
            transit_index += 1
            if transit.predictions_count == 0:
                continue

            # Model information
            transit_flux = transit.predictions[0].f_mean[:, 0]
            exoplanet = transit.light_curve.exoplanets[0]
            transit_name = f"{transit.light_curve.star_name}_{exoplanet.name}"

            # Fill the gaps in the light curve
            lc = deepcopy(original_lc)
            lc.time, lc.flux = lc.fill_gaps()
            lc.flux = np.nan_to_num(lc.flux, nan=np.nanmean(lc.flux))

            # Make the transit odd length; this makes the indexing in the calculation easier
            if len(transit_flux) % 2 == 0:
                transit_flux = transit_flux[:-1]

            # Calculate the correlation and threshold
            corr = calculate_correlation(lc.flux, transit_flux)
            gap, ref = compute_gap_ref(exoplanet, lc, gap, ref)
            threshold = np.array(moving_avg_gap(corr, gap, ref, bias))

            # Cut the light curve and correlation to the threshold's length.
            # The light curve is the longest, the correlation is cut by the length of the transit model,
            # the threshold is cut by the gap and ref parameters.
            corr_padding = len(transit_flux) // 2
            t_padding = gap + ref

            lc.time = lc.time[corr_padding + t_padding : -(corr_padding + t_padding)]
            lc.flux = lc.flux[corr_padding + t_padding : -(corr_padding + t_padding)]

            corr = corr[t_padding:-t_padding]

            # Record results
            if results_path is not None:
                # Find the detected transits and identify correctly detected transits
                # (only if the light curve has exoplanets)
                detectable_transit_times = []

                # Compare the found transits with the correct ones
                if compare_existing:
                    # Find detectable transits based on the nans in the filled light curve
                    # If the transit time is surrounded by nans in flux, it is not detectable
                    new_lc = deepcopy(original_lc)
                    new_lc.time, new_lc.flux = new_lc.fill_gaps()
                    new_lc.time = new_lc.time[
                        corr_padding + t_padding : -(corr_padding + t_padding)
                    ]

                    new_lc.flux = new_lc.flux[
                        corr_padding + t_padding : -(corr_padding + t_padding)
                    ]

                    detectable_transit_times = get_transit_times(
                        new_lc, new_lc.exoplanets
                    )

                # Detect transits
                transit_times, corr_threshold, n_ok_detected = detect_transits(
                    lc.time,
                    corr,
                    threshold,
                    detectable_transit_times,
                )

                # If the found transits are to be evaluated against the correct transits
                if compare_existing:
                    detectable.append(len(detectable_transit_times))
                    ok_detected.append(n_ok_detected)

                if corr_threshold[0] >= 1:
                    detected.append(len(transit_times))
                else:
                    detected.append(0)

                corr_results["t_positions"][transit_name] = transit_times
                corr_results["corr_threshold"][transit_name] = corr_threshold

            # Plot the correlation plots
            if plot_path is not None:
                plot_correlation(
                    lc,
                    corr,
                    threshold,
                    plot_path / f"{transit_name}/",
                    file_format,
                )

        # After the light curve is processed for all transit models
        # Save the results to file
        if results_path is not None:
            # The columns are light curve info and transit models
            columns = create_results_columns(transit_models)

            if compare_existing:
                save_results(
                    ok_detected,
                    results_path / f"ok_detected.csv",
                    columns,
                    mode="a",
                )
                save_results(
                    detectable,
                    results_path / f"detectable.csv",
                    columns,
                    mode="a",
                )

            save_results(
                detected,
                results_path / f"detected.csv",
                columns,
                mode="a",
            )

            save_results_json(
                corr_results,
                results_path / f"corr_results/corr_results_lc={original_lc.config_str}.json",
            )


def calculate_correlation(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Calculate correlation between two signals. Correlation is z-score normalized.
    Values below 1 are set to 1.

    Args:
        s1 (np.ndarray): 1D signal.
        s2 (np.ndarray): 1D signal

    Returns:
        np.ndarray: computed correlation values.
    """
    corr = np.correlate(s1, s2)
    corr_norm = (corr - np.mean(corr)) / np.std(corr)
    corr_norm[corr_norm < 1] = 1
    return corr_norm


def get_transit_times(lc: "LightCurve", exoplanets: List["Exoplanet"]) -> List[float]:
    """Returns detectable transit times for light curve.

    Args:
        lc ("LightCurve"): light curve.
        exoplanets (List["Exoplanet"]): list of exopalnets.

    Returns:
        List[float]: transit times.
    """
    # Get the transit times and flatten them
    groups = [lc.get_transits(e) for e in exoplanets]
    flat = [time for group in groups for time in group]

    # Check if the transit is surrounded by nans in flux.
    transits = []
    for transit in flat:
        # Index of the transit in data
        index = sum(transit >= lc.time)
        # Look around
        margin = 5
        if index - margin < 0:
            margin = margin - index - 1
        flux_range = lc.flux[index - margin : index + margin]
        # If there are nans in the surrounding, do not include it
        if len(flux_range) == 0 or np.isnan(np.min(flux_range)):
            pass
        else:
            # If there are no nans, include it
            transits.append(transit)

    return transits


def compute_gap_ref(
    e: "Exoplanet",
    lc: "LightCurve",
    gap: int,
    ref: int,
    gap_coeff: float = 1.5,
    ref_coeff: float = 6,
) -> Tuple[float, float]:
    """_summary_

    Args:
        e (Exoplanet): exoplanet.
        lc (LightCurve): light curve.
        gap (int): existing gap value. Computed if None.
        ref (int): existing ref value. Computed if None
        gap_coeff (float, optional): multiplicator for the transit duration in samples used for computing the gap value. Defaults to 1.5.
        ref_coeff (float, optional): multiplicator for the transit duration in samples used for computing the ref value. Defaults to 6.

    Returns:
        Tuple[float, float]: new gap and ref values.
    """
    # Convert the transit duration to samples
    transit_duration_in_samples = e.transit_duration_h * 60 * 60 / lc.cadence_period

    # Compute the gap and ref values
    if gap is None:
        gap = int(gap_coeff * transit_duration_in_samples)
    if ref is None:
        ref = int(ref_coeff * transit_duration_in_samples)

    return (gap, ref)


def plot_correlation(
    lc: "LightCurve",
    corr: np.ndarray,
    threshold: np.ndarray,
    plot_path: Path,
    file_format: str = "png",
) -> None:
    """Plot the correlation.

    Args:
        lc (LightCurve): light curve.
        corr (np.ndarray): correlation values.
        threshold (np.ndarray): threshold values.
        plot_path (Path): directory for the figure.
        file_format (str, optional): file format for the figure. Defaults to "png".
    """
    plot_path.mkdir(parents=True, exist_ok=True)

    # Create the axis
    _, ax = plt.subplots()

    # Plot the light curve
    ax.scatter(
        lc.time,
        lc.flux,
        s=0.5,
        edgecolors="none",
        color="black",
    )

    # Plot exoplanet transits
    lc._plot_exoplanets(ax)

    # Set title and text
    ax.set_title(lc.star_name, pad=15)
    ax.text(
        0.5,
        1.02,
        f"{lc.mission}, {lc.author}, c.p.: {lc.cadence_period}, {lc.batch_name}: {', '.join(map(str, lc.batch))}",
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=8,
    )

    # Plot correlation
    ax.plot(
        lc.time,
        corr,
        color="tab:red",
        linewidth=0.5,
    )

    # Plot threshold
    ax.plot(
        lc.time,
        threshold,
        linewidth=0.5,
        color="blue",
        label="CFAR Threshold",
    )

    # Set labels
    ax.set_xlabel(lc.xlabel)
    ax.set_ylabel("Correlation amplitude")

    # Mark detected transits
    plot_transits(ax, lc.time, corr, threshold)

    plt.tight_layout()

    # Save figure
    plt.savefig(plot_path / f"{lc.config_str}_DET.{file_format}")

    # Close figure
    plt.close()


def plot_transit(ax: plt.axis, start: float, end: float) -> None:
    """Plot a single detected transit.

    Args:
        ax (plt.axis): axis.
        start (float): start in time.
        end (float): end in time.
    """
    ax.axvspan(
        start - 0.1,
        end + 0.1,
        edgecolor="none",
        facecolor="#ffd23d",
        alpha=0.5,
    )


def plot_transits(
    ax: plt.axis, time: np.ndarray, corr: np.ndarray, threshold: np.ndarray
) -> None:
    """Plot detected transits.

    Args:
        ax (plt.axis): axis.
        time (np.ndarray): time values.
        corr (np.ndarray): correlation values.
        threshold (np.ndarray): threshold values.
    """

    # Mask where correlation exceeds threshold
    mask = corr > threshold
    # Index for iterating the mask
    i = 0
    # Transit start
    t_start = None
    while i < len(mask):
        # If the correlation exceeds the threshold
        if mask[i]:
            # And not already in transit
            if t_start is None:
                # Mark start of transit
                t_start = time[i]
        else:
            # If the correlation is below the threshold
            if t_start is not None:
                # And in transit
                # Plot the transit
                plot_transit(ax, t_start, time[i])
                # End the transit
                t_start = None
                # And jump off
                i += 60
        i += 1


def create_results_columns(transit_models: List["Model"]) -> List[str]:
    """Create columns for the results.

    Args:
        transit_models (List[Model]): list of transit models.

    Returns:
        List[str]: column names.
    """
    columns = ["star_name", "mission", "author", "cadence_period", "batch"]
    # Add transit names
    columns.extend(
        [
            f"{transit.light_curve.star_name} ({transit.light_curve.exoplanets[0].name})"
            for transit in transit_models
        ]
    )
    return columns


def save_results(
    results: List[Any], filepath: Path, columns: List[str], mode: str = "w"
) -> None:
    """Save results to csv file.

    Args:
        results (List[Any]): results.
        filepath (Path): csv filename.
        columns (List[str]): columns describing the results
        mode (str, optional): write (w) or append to the file (a). Defaults to "w".
    """
    df = pd.DataFrame(results).transpose()

    df.columns = columns
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, mode=mode, header=not filepath.exists())


def save_results_json(results: Dict[str, Any], filepath: Path):
    """Save dictionary as json.

    Args:
        results (Dict[str, Any]): results.
        filepath (Path): JSON filename.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump([results], f, indent=4)


def detect_transits(
    time: np.ndarray,
    corr: np.ndarray,
    threshold: np.ndarray,
    correct_transits: List[float],
) -> Tuple[List[float], List[float], int]:
    """Detect transits using correlation and threshold.

    Args:
        time (np.ndarray): time values.
        corr (np.ndarray): correlation values.
        threshold (np.ndarray): threshold values
        correct_transits (List[float]): correct transit times.

    Returns:
        Tuple[List[float], List[float], int]: list of detected transit times,
        max(corr)/mean(threshold) for these times, and number of detected
        transits which match known transits. If no transit is detected, the largest
        max(corr)/mean(threshold) with it time positions is used.
    """

    mask = corr > threshold  # Mask where correlation exceeds threshold
    t_start = None  # Transit start time
    ok_detected = 0  # How many transits correctly detected
    i = 0  # Position in the mask
    t_pos = []  # Transit times positions
    c_t = []  # Max correlation / mean threshold for detected transits
    t_i_start = None  # Index of the start of the transit in the data
    largest = 0  # Largest correlation / threshold value
    largest_time = 0  # Time for the largest correlation / threshold value

    # Iterate over the mask
    while i < len(mask):
        # Value of the correlation / threshold in the point
        point_corr_th = corr[i] / threshold[i]
        if point_corr_th > largest:
            largest = point_corr_th
            largest_time = time[i]

        # If the correlation exceeds the threshold
        if mask[i]:
            # And not in transit
            if t_start is None:
                # Store start of transit position
                t_start = time[i]
                t_i_start = i
        else:
            # If the correlation is below the threshold
            # And in transit
            if t_start is not None:
                # Save the mean transit position
                t_pos.append(np.mean([t_start, time[i]]).item())
                # Check for known transits
                if transit_in_region(t_start, time[i], correct_transits):
                    ok_detected += 1

                t_start = None

                # Get the max correlation / mean threshold in the region of the transit
                c_max = np.max(corr[t_i_start:i]).item()
                t_mean = np.mean(threshold[t_i_start:i]).item()
                c_t.append(c_max / t_mean)
        i += 1

    # If no transits were found, put the largest corr/threshold as the result
    if len(c_t) == 0:
        t_pos = [largest_time.item()]
        c_t = [largest.item()]

    return t_pos, c_t, ok_detected


def transit_in_region(start: float, end: float, transits: List[float]) -> bool:
    """Check whether a transit is in the provided region.

    Args:
        start (float): start in time.
        end (float): end in time.
        transits (List[float]): list of transit times.

    Returns:
        bool: True if transit in the region.
    """
    for transit in transits:
        if start < transit < end:
            return True
    return False
