from exo_search.modeling.manager import Manager
from exo_search.entities.light_curve import LightCurve, load_lcs_from_dir_list
from pathlib import Path
import pandas as pd
from typing import List


def main(
    model_path: Path,
    detrended_lc_path: Path,
    diff_threshold: float,
    lc_path: Path = None,
    prediction_path: Path = None,
    results_path: Path = None,
    n: int = None,
) -> None:
    """Detrending. Subtract predicted flux from the original flux.

    Args:
        model_path (Path): directory with models.
        detrended_lc_path (Path): directory for the new light curves.
        diff_threshold (float): threshold for the mean absolute difference.
        lc_path (Path, optional): alternative directory with light curves. If provided, the light curves will be used for the detrending. By default, light curve attached to the model is used.
        prediction_path (Path, optional): path to the predictions. Defaults to <model_path>/predictions/.
        results_path (Path, optional): directory where to save the results (which light curve was detrended). Defaults to None.
        n (int, optional): load only the n-th model from the directory. All models are loaded by default.
    """

    if prediction_path is None:
        prediction_path = model_path / "predictions/"

    # Load models and predictions
    manager = Manager()
    manager.load(model_path, n)
    manager.load_predictions(prediction_path)

    # Filter for only models with predictions
    print(f"Number of models before filtering {len(manager.models)}")
    manager.models = manager.filter_models(lambda x: x.predictions_count > 0)
    print(f"Number of models after filtering {len(manager.models)}")

    # Dictionary for the results
    results = {}

    # Iterate over the models
    for m in manager.models:
        print(f"Detrending {m.config_str}")

        lc = None

        # If no alternative path to light curves is provided,
        # use the one attached to the model
        if lc_path is None:
            lc = m.light_curve
        else:
            # Load alternative light curves
            # Name of the file JSON file with the light curve must match the light curve info
            lcs = load_lcs_from_dir_list(
                lc_path,
                [m.light_curve.info],
            )
            # Else try to find it in the list
            for lc in lcs:
                if m.light_curve.matches(
                    lc.star_name, lc.mission, lc.author, lc.cadence_period, lc.batch
                ):
                    print("Found original lc!")
                    break
            else:
                print(f"Failed to get original light curve. Skipping {m.config_str}")
                continue

        # Find prediction with the same x values
        for prediction in m.predictions:
            if prediction.matches(lc.time[:, None]):
                break
        else:
            print(
                f"No prediction (out of {m.predictions_count}) matches the used light curve. Skipping {m.config_str}"
            )
            continue

        # Calculate the difference between the old and new flux values
        diff = lc.flux - prediction.f_mean[:, 0]
        diff_mean_abs = sum(abs(diff)) / len(diff)

        # Compare the difference with the threshold
        if diff_mean_abs >= diff_threshold:
            # Create new light curve and save it

            new = LightCurve(
                lc.star_name,
                lc.time,
                lc.flux - prediction.f_mean[:, 0],
                lc.mission,
                lc.author,
                lc.cadence_period,
                lc.batch,
                exoplanets=lc.exoplanets,
                config_str=lc.config_str,
            )

            new.save_to_file(detrended_lc_path / f"{new.config_str}.json")
        else:
            # Else save the old light curve
            lc.save_to_file(detrended_lc_path / f"{lc.config_str}.json")

        print("Logging results")
        log_results(results, bool(diff_mean_abs >= diff_threshold), lc)

    # Save the results
    if results_path is not None:
        results_path.mkdir(parents=True, exist_ok=True)
        save_results(
            results,
            results_path / f"detrending_lc={n if n is not None else 'all'}.csv",
            ["star_name", "mission", "author", "cadence", "batch", "detrended"],
        )


def log_results(results, result, lc: LightCurve):
    """Log results with light curve info as key."""
    results[lc.csv_info] = result


def save_results(results, filepath: Path, columns: List[str]):
    """Save found results to a file.

    Args:
        results: results to be saved.
        filepath (Path): file path for the results.
        columns (List[str]): column names.
    """
    if filepath is None:
        return

    # Produce data for saving
    to_save = []
    for key, item in results.items():
        record = key.split(",")
        record.append(item)
        to_save.append(record)

    df = pd.DataFrame(to_save)

    if df.empty:
        print("Results are empty!")
        return

    df.columns = columns
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
