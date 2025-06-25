"""Load models from directory and train them.
Save them to the original directory or a new one."""

from exo_search.modeling.manager import Manager
from pathlib import Path


def main(model_path: Path, n: int = None, save_model_path: Path = None) -> None:
    """Load Model objects from a directory and train them. Save the trained models
    to their original directory or a new one.

    Args:
        model_path (Path): directory with models.
        n (int, optional): load n-th model from the directory. Loads all models by default.
        save_model_path (Path, optional): directory for new models. Models are saved to their
        original directory by default.
    """
    if save_model_path is None:
        save_model_path = model_path
    manager = Manager()
    manager.load(model_path, n)
    manager.train(save_model_path)
