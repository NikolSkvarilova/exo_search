from exo_search.utils.plot_difference import plot_difference
from exo_search.modeling.manager import Manager
from exo_search.entities.light_curve import LightCurve, load_lcs_from_dir_list
import numpy as np
from pathlib import Path


def main(
    model_path: Path,
    plot_path: Path = None,
    plot_attr_path: Path = None,
    lc_path: Path = None,
    prediction_path: Path = None,
    file_format: str = "png",
    label_path: Path = None,
    n: int = None,
) -> None:
    """Plot the original light curve, the detrending model, the difference, and the detrended light curve.
    Also produce attributes plots.

    Args:
        model_path (Path): directory with the models.
        plot_path (Path, optional): directory for the plots. If not provided, no plots for the individual light curves will be produced.
        plot_corr_path (Path, optional): directory for the correlation plots. If not provided, no correlation plots will be produced.
        lc_path (Path, optional): directory with light curves used as the base for detrending. The model's light curve is used by default.
        prediction_path (Path, optional): directory with predictions. Defaults to <model_path>/predictions/.
        file_format (str, optional): file format for the plots. Defaults to "png".
        label_path (Path, optional): path to labels for the stars. If not provided, the correlation plots will not be colored.
    """
    if prediction_path is None:
        prediction_path = model_path / "predictions"

    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)

    # Load models and predictions
    manager = Manager()
    manager.load(model_path, n)
    manager.load_predictions(prediction_path)

    # Filter out models without a prediction
    print(f"Number of models before filtering {len(manager.models)}")
    manager.models = manager.filter_models(lambda x: x.predictions_count > 0)
    print(f"Number of models after filtering {len(manager.models)}")

    # Plot detrending results
    models_to_plot = []

    # Iterate over the models
    for m in manager.models:
        lc = None
        if lc_path is None:
            lc = m.light_curve
        else:
            lcs = load_lcs_from_dir_list(
                lc_path,
                [m.light_curve.info],
            )
            for lc in lcs:
                if m.light_curve.matches(
                    lc.star_name, lc.mission, lc.author, lc.cadence_period, lc.batch
                ):
                    print("Found original lc!")
                    break
            else:
                print(f"Failed to get original light curve. Skipping {m.config_str}")
                continue

        for prediction in m.predictions:
            if prediction.matches(lc.time[:, None]):
                break
        else:
            print(f"Failed to load prediction. Skipping {m.config_str}")
            continue

        # Create new light curve
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

        # Assign the model detrending attributes
        if plot_attr_path is not None:
            diff = lc.flux - new.flux
            m.diff_sum = sum(abs(diff))
            m.diff_mean_abs = sum(abs(diff)) / len(diff)
            m.diff_std = np.std(diff)
            m.diff_var = np.var(diff)
            m.diff_mean = np.mean(diff)
            m.diff_median = np.mean(diff)
            models_to_plot.append(m)

        # Plot individual plots
        if plot_path is not None:
            new.plot(
                save_fig=plot_path / f"{new.config_str}_DETRENDED.{file_format}",
                show_fig=False,
            )
            lc.plot(
                save_fig=plot_path / f"{lc.config_str}_ORIGINAL.{file_format}",
                show_fig=False,
            )

            m.plot(
                prediction,
                show_fig=False,
                save_fig=plot_path / f"{new.config_str}_MODEL.{file_format}",
            )

            plot_difference(
                lc.time,
                lc.flux,
                new.flux,
                show_fig=False,
                save_fig=plot_path / f"{new.config_str}_DIFFERENCE.{file_format}",
                x_label=lc.xlabel,
            )

    # Plot attribute plots
    if plot_attr_path is not None:
        plot_attr_path.mkdir(parents=True, exist_ok=True)
        manager.models = models_to_plot

        # Color the models
        if label_path is not None:
            mapping = {
                "major_activity": "red",
                "minor_activity": "orange",
                "clear_transit": "green",
                "noisy": "blue",
            }
            manager.color_models_by_star(label_path, mapping)

        manager.plot_model_attributes(
            save_fig=plot_attr_path,
            model_attributes=[
                "likelihood",
                "likelihood_var",
                "diff_sum",
                "diff_std",
                "diff_var",
                "diff_mean",
                "diff_mean_abs",
                "diff_median",
            ],
            prediction_attributes=[],
            operations=[],
            kernel_attributes=[],
            colored_models=label_path is not None,
            file_format=file_format,
        )
