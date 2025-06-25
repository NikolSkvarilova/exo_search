from exo_search.modeling.manager import Manager
from exo_search.modeling.model import Model
from pathlib import Path

def main(model_path: Path, dst_path: Path, index: int = 1, n: int = None) -> None:
    """Extract kernel from an existing model. Put it into a new model and save it.

    Args:
        model_path (Path): directory with the models.
        dst_path (Path): directory for the new models.
        index (int, optional): index of the kernel. Stars at 0. Defaults to 1.
        n (int, optional): load n-th model from the directory. Loads all models by default.
    """
    manager = Manager()
    manager.load(model_path, n)

    new_manager = Manager()
    for model in manager.models:
        # Extract kernel
        kernel = model.model.kernel.kernels[index]

        # Create new model
        new_model = Model(
            model.light_curve,
            model.config_str,
            model.training_time,
            model.period,
            model.trained,
            model.likelihood,
            model.likelihood_var,
            kernel,
        )

        new_manager.models.append(new_model)

    new_manager.save(dst_path)