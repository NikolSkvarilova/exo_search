from exo_search.modeling.manager import Manager
from exo_search.entities.light_curve import load_lcs_from_dir_list
from pathlib import Path
import numpy as np


def main(
    model_path: Path,
    prediction_path: Path = None,
    lc_path: Path = None,
    n: int = None,
    trained_only: bool = True,
    gap_s: float = None,
    predict_y: bool = False,
) -> None:
    """Create predictions for models.

    Args:
        model_path (Path): directory with models.
        prediction_path (Path, optional): directory with predictions. Defaults to <model_path>/predictions/.
        lc_path (Path, optional): directory with light curves. The prediction will be carried out over the light curve's time. Model's light curve is used by default.
        n (int, optional): load n-th model from the directory. All models are loaded by default.
        trained_only (bool, optional): filter for trained models. Defaults to True.
        gap_s (float, optional): if provided, the prediction is carried out over an evenly spaced time. Original time values are used by default.
        predict_y (bool): if True, the y-value are predicted as well. Only the f-values are predicted by default. Defaults to False.
    """
    if prediction_path is None:
        prediction_path = model_path / "predictions"

    # Load models
    manager = Manager()
    manager.load(model_path, n)
    manager.load_predictions(prediction_path)

    # Filter out non-trained models
    if trained_only:
        manager.models = manager.filter_models(lambda x: x.trained)

    # Iterate over the models
    for m in manager.models:
        x_values = None
        # If light curves come from a different directory, find the one corresponding to the model
        if lc_path:
            # Load all light curves with the same info
            lcs = load_lcs_from_dir_list(lc_path, [m.light_curve.info])
            # Find the one matching the model's light curve
            for lc in lcs:
                if m.light_curve.matches(
                    lc.star_name, lc.mission, lc.author, lc.cadence_period, lc.batch
                ):
                    print("Found lc!")
                    break
            else:
                print(f"Failed to find light curve for {m.light_curve.star_name}")
                continue

            x_values = lc.time[:, None]
        else:
            # Or use the model's light curve
            x_values = m.light_curve.time[:, None]

        # Create custom x values based on provided gap
        if gap_s is not None:
            start = x_values[0]
            end = x_values[-1]
            step = gap_s / (24 * 60 * 60)
            x_values = np.arange(start, end + step, step)[:, None]

        print("Predicting")
        # Make prediction
        m.predict(x_values, predict_y=predict_y)
        # Save prediction
        m.save_predictions(prediction_path / f"{m.config_str}.json")
