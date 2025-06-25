from exo_search.modeling.manager import Manager
from pathlib import Path


def main(model_path: Path, prediction_path: Path = None) -> None:
    """Print out various counts for Model objects from a directory.

    Args:
        model_path (Path): directory with the models.
        prediction_path (Path, optional): directory with the predictions. Defaults to <model_path>/predictions/.
    """
    if prediction_path is None:
        prediction_path = model_path / "predictions"

    # Load models and predictions
    manager = Manager()
    manager.load(model_path)
    manager.load_predictions(prediction_path)

    # Calculate counts
    total = len(manager.models)
    trained = len(manager.filter_models(lambda x: x.trained))
    with_prediction = len(manager.filter_models(lambda x: len(x.predictions) > 0))

    # Print the results
    print(f"Total number of models: {total}")
    print(f"Trained: {trained}")
    print(f"With at least 1 prediction: {with_prediction}")
