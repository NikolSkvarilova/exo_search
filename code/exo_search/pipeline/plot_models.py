from exo_search.modeling.manager import Manager
from pathlib import Path


def main(
    model_path: Path,
    plot_path: Path,
    prediction_path: Path = None,
    file_format: str = "png",
    n: int = None,
    is_transit: bool = False,
) -> None:
    """Plot models' predictions. Models without predictions are ignored.

    Args:
        model_path (Path): directory with the models.
        plot_path (Path): directory for the plots.
        prediction_path (Path, optional): directory with predictions. Defaults to <model_path>/predictions/.
        file_format (str, optional): file format for the plots. Defaults to "png".
        n (int): plot only the n-th model from the directory. All models from the directory are plotted by default.
        is_transit (bool): if True, the x-axis is plotted in hours.
    """
    if prediction_path is None:
        prediction_path = model_path / "predictions"

    # Load models and predictions
    manager = Manager()
    manager.load(model_path, n)
    manager.load_predictions(prediction_path)

    # Filter out models without predictions
    print(f"Number of models before filtering: {len(manager.models)}")
    manager.models = manager.filter_models(lambda x: x.predictions_count > 0)
    print(f"Number of models after filtering: {len(manager.models)}")

    # Plot predictions
    manager.plot_predictions(
        show_fig=False,
        save_fig=plot_path,
        plot_exoplanets=True,
        file_format=file_format,
        is_transit=is_transit,
    )
