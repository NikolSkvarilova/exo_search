from exo_search.modeling.manager import Manager
from pathlib import Path


def main(
    model_path: Path,
    attr: str,
    comparator: str,
    value: float,
    new_model_path: Path,
    prediction_path: Path = None,
    new_prediction_path: Path = None,
    n: int = None,
) -> None:
    """Filter models by value.

    Args:
        model_path (Path): directory.
        attr (str): attribute of the model
        comparator (str): =, >, <, !=.
        value (int): value to compare the attribute with.
        new_model_path (Path): directory for models matching the condition.
        prediction_path (Path, optional): directory with predictions. Defaults to None.
        n (int, optional): load only n-th model from the directory. All models are loaded by default.
    """

    if prediction_path is None:
        prediction_path = model_path / "predictions"

    if new_prediction_path is None:
        new_prediction_path = new_model_path / "predictions"

    # Load models and predictions
    manager = Manager()
    manager.load(model_path, n)
    manager.load_predictions(prediction_path)

    print(f"Number of models before: {len(manager.models)}")

    # Filter models
    if comparator == "=":
        manager.models = manager.filter_models(lambda x: getattr(x, attr) == value)
    elif comparator == "<":
        manager.models = manager.filter_models(lambda x: getattr(x, attr) < value)
    elif comparator == ">":
        manager.models = manager.filter_models(lambda x: getattr(x, attr) > value)
    elif comparator == "!=":
        manager.models = manager.filter_models(lambda x: getattr(x, attr) != value)


    print(f"Number of models after: {len(manager.models)}")

    # Save predictions
    manager.save_predictions(new_prediction_path)

    # Save models
    manager.save(new_model_path)
