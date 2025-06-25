from exo_search.modeling.manager import Manager
from pathlib import Path


def main(
    model_path: Path,
    plot_corr_path: Path,
    file_format: str = "png",
    prediction_path: Path = None,
    label_path: Path = None,
) -> None:
    """Plot model attributes.

    Args:
        model_path (Path): directory with the models.
        plot_corr_path (Path): directory for the plots.
        file_format (str, optional): file format for the plots. Defaults to "png".
        prediction_path (Path, optional): directory with the predictions. Defaults to <model_path>/predictions/.
        label_path (Path, optional): path to labels for the stars. If not provided, the plots will not be colored.
    """

    if prediction_path is None:
        prediction_path = model_path / "predictions"

    # Load models and predictions
    manager = Manager()
    manager.load(model_path)
    manager.load_predictions(prediction_path)

    # Filter out models without predictions
    print("Before filtering: ", len(manager.models))
    manager.models = manager.filter_models(lambda x: x.predictions_count > 0)
    print("After filtering: ", len(manager.models))

    # Color the models
    if label_path is not None:
        mapping = {
            "major_activity": "red",
            "minor_activity": "orange",
            "clear_transit": "green",
            "noisy": "blue",
        }
        manager.color_models_by_star(label_path, mapping)

    # Plot model attributes
    manager.plot_model_attributes(
        save_fig=plot_corr_path,
        colored_models=label_path is not None,
        file_format=file_format,
    )
