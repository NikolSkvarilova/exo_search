from exo_search.modeling.model import Model
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Dict
import json
import matplotlib.patches as mpatches
import gpflow
from pathlib import Path
from exo_search.utils.logger import Logger, Category
from exo_search.entities.light_curve import LightCurve
import collections


class Manager:
    """Holds a list of Model objects."""

    logger = Logger("manager.py")

    def __init__(self) -> None:
        """Constructor."""
        self.models: list[Model] = []

    def train(self, save_models: Path) -> None:
        """Train models. After every training, save the model a directory.

        Args:
            save_models (Path): directory.
        """
        for model in self.models:
            print(f"Training {model.config_str}")
            model.train()
            if save_models:
                model.save_to_file(
                    fn_data=save_models / f"{model.config_str}.json",
                    fn_model=save_models / f"gpflow_models/{model.config_str}.pickle",
                )

    def predict(self, save_dir: Path) -> None:
        """Make predictions for the models. After every predicting, save the prediction to a directory.

        Args:
            save_dir (Path): directory.
        """
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        for model in self.models:
            model.predict(model.x)
            if save_dir is not None:
                model.save_predictions(save_dir / f"{model.config_str}.json")

    def plot_predictions(
        self,
        show_fig: bool = True,
        save_fig: Path = None,
        plot_exoplanets: bool = True,
        file_format: str = "png",
        plot_exoplanet_info: bool = False,
        is_transit: bool = False,
    ) -> None:
        """Plot predictions.

        Args:
            show_fig (bool, optional): display the figure. Defaults to True.
            save_fig (Path, optional): directory to save the figure to. Defaults to None.
            plot_exoplanets (bool, optional): plot transits. Defaults to True.
            file_format (str, optional): file format for the saved figure. Defaults to "png".
            plot_exoplanet_info (bool, optional): plot the exoplanet periods, etc. Defaults to False.
            is_transit (bool, optional): if True, the x-axis is in hours. Defaults to False.
        """

        if save_fig is not None:
            save_fig.mkdir(parents=True, exist_ok=True)

        for model in self.models:
            # If the model has no prediction, skip it
            if len(model.predictions) == 0:
                continue
            # Construct extra title (displayed under title)
            extra_title = [""]
            extra_title.append(
                f"likelihood: {round(model.likelihood, 2) if model.likelihood is not None else '-'}"
            )

            extra_title = ", ".join(extra_title)

            extra_padding = 0

            # Add exoplanet info into the extra title
            if plot_exoplanet_info and len(model.light_curve.exoplanets) > 0:
                extra_title += "\n"
                extra_padding += 10
                periods = []
                for exoplanet in model.light_curve.exoplanets:
                    if exoplanet.period is not None and model.period is not None:
                        periods.append(
                            f"{exoplanet.name[0]} ({round(model.period / exoplanet.period, 2)})"
                        )
                    else:
                        periods.append(f"{exoplanet.name[0]} (-)")

                extra_title += ", ".join(periods)

            # Plot the model
            model.plot(
                model.predictions[-1],
                show_fig=show_fig,
                save_fig=(
                    save_fig / f"{model.config_str}.{file_format}"
                    if save_fig is not None
                    else None
                ),
                plot_exoplanets=plot_exoplanets,
                extra_title=extra_title,
                extra_padding=extra_padding,
                is_transit=is_transit,
            )

    def save(self, dir: Path) -> None:
        """Saves models into directory.

        Args:
            dir (Path): directory.
        """
        for model in self.models:
            model.save_to_file(
                fn_data=dir / f"{model.config_str}.json",
                fn_model=dir / f"gpflow_models/{model.config_str}.pickle",
            )

    def create_models_from_lc(
        self,
        lcs: List[LightCurve],
        kernel: gpflow.kernels.Kernel,
        kernel_name: str,
        model_path: Path = None,
        create_missing=True,
    ) -> None:
        """Create Model objects from provided light curves. Also checks for already existing models.

        Args:
            lcs (List[LightCurve]): light curves.
            kernel (gpflow.kernels.Kernel): kernel.
            kernel_name (str): kernel string representation.
            model_path (Path, optional): directory to try to load models from. Defaults to None.
            create_missing (bool, optional): if True, creates new models. Defaults to True.
        """
        for lc in lcs:
            # Check if this model exists
            model = None
            if model_path is not None:
                # Load the model
                model = Model.load_from_file(
                    fn_data=model_path / f"{lc.config_str},{kernel_name}.json",
                    fn_model=model_path
                    / f"gpflow_models/{lc.config_str},{kernel_name}.pickle",
                )

                # If success, add it to the list and continue, else create it
                if model is not None:
                    self.models.append(model)
                    continue

            # Create new model
            if create_missing:
                model = Model(
                    lc,
                    f"{lc.config_str},{kernel_name}",
                    kernel=gpflow.utilities.deepcopy(kernel),
                )

            if model is not None:
                self.models.append(model)

    def load(self, dir: Path, n: int = None) -> None:
        """Loads models from directory.

        Args:
            dir (Path): directory to load models from.
            n (int, optional): index of the JSON file to load. All JSON files are loaded by default. Stars from 1.
        """

        # List of filenames
        filenames = [item.name for item in dir.iterdir() if item.suffix == ".json"]
        self.logger.log(
            Category.INFO,
            "",
            "load()",
            f"{len(filenames)} models available in directory {dir}",
        )

        # Load only the n-th model or none
        if n is not None:
            if 0 < n <= len(filenames):
                filenames = [filenames[n - 1]]
            else:
                return

        loaded = 0
        # Load models from files
        for fn in filenames:
            fn = fn[: (-1) * len(".json")]
            print(f"Loading {fn}")
            model = Model.load_from_file(
                dir / f"{fn}.json", dir / f"gpflow_models/{fn}.pickle"
            )
            if model is not None:
                loaded += 1
                self.models.append(model)

        self.logger.log(
            Category.INFO, "", "load()", f"loaded {loaded} models from directory {dir}"
        )

    def filter_models(self, condition: Callable) -> List[Model]:
        """Filter models based on a condition

        Args:
            condition (Callable): lambda function specifying condition.

        Returns:
            List[Model]: list of Model objects.
        """

        return [model for model in self.models if condition(model)]

    def color_models_by_star(
        self, label_path: Path, mapping: Dict[str, str], no_label_color: str = "none"
    ) -> None:
        """Add color and label attributes to all models.

        Args:
            label_path (Path): directory with labels.
            mapping (Dict[str, str]): dictionary with mapping label to color.
            no_label_color (str, optional): color for models which do not match any label. Defaults to "none".
        """

        # Iterate over all models
        for model in self.models:
            star_name = model.light_curve.star_name

            # Add default color and label
            model.color = no_label_color
            model.label = "No label"

            # Try to load the label
            try:
                with open(label_path / f"{star_name}.json") as f:
                    data = json.load(f)
            except OSError as e:
                continue
            except json.JSONDecodeError as e:
                self.logger.log(
                    Category.ERROR,
                    label_path / f"{star_name}.json",
                    "color_models_by_lc()",
                    f"Bad JSON ({e})",
                )
                continue

            star_labels = data["flags"]

            # Iterate over the mapping and assign the first positive color and label
            for flag, color in mapping.items():
                value = star_labels.get(flag, None)
                if value is None:
                    print(f"WARNING: {flag} missing in labels.")
                    continue

                if value:
                    model.label = flag
                    model.color = color
                    break

    def save_predictions(self, path: Path) -> None:
        """Saves all models' predictions into directory.

        Args:
            path (Path): Directory to save the predictions to.
        """
        for model in self.models:
            model.save_predictions(path / f"{model.config_str}.json")

    def load_predictions(self, path: Path) -> None:
        """Loads all models' predictions from the directory.

        Args:
            path (Path): Directory to load the predictions from.
        """
        for model in self.models:
            model.load_predictions(path / f"{model.config_str}.json")

    def plot_model_attributes(
        self,
        save_fig: Path,
        model_attributes: List[str] = None,
        prediction_attributes: List[str] = None,
        operations: List[callable] = None,
        kernel_attributes: bool = True,
        colored_models: bool = False,
        file_format: str = "png",
    ) -> None:
        """Creates figures of various model attributes plotted against each other.

        Args:
            save_fig (Path): directory.
            model_attributes (List[str], optional): list of attributes of the Model object. Defaults to ["likelihood"].
            prediction_attributes (List[str], optional): list of attribtues for the last prediction of the models. Defaults to ["f_mean", "f_var", "y_var", "f_2s", "y_2s"].
            operations (List[callable], optional): operations to perform over the prediction attributes. Defaults to [np.mean, np.median, np.min, np.max, np.std, np.var].
            kernel_attributes (bool, optional): list of attributes of the kernel. Defaults to whatever is in the gpflow.utilities.parameter_dict of the first model.
            colored_models (bool, optional): if True, it is expected that the model has a color and label attributes that are used in the figure. Defaults to False.
            file_format (str, optional): file format for the figure. Defaults to "png".
        """

        if len(self.models) == 0:
            return

        save_fig.mkdir(parents=True, exist_ok=True)

        # Set the default values for the lists
        if model_attributes is None:
            model_attributes = ["likelihood", "likelihood_var"]

        if prediction_attributes is None:
            prediction_attributes = ["f_mean", "f_var", "y_var", "f_2s", "y_2s"]

        if operations is None:
            operations = [np.mean, np.median, np.min, np.max, np.std, np.var]

        # Create dictionary with attribute (+ operation)
        # as a key and values for all models

        # Dictionary with all values for all models
        models_dict = collections.defaultdict(list)

        # Add model attributes
        for attr in model_attributes:
            models_dict[attr] = [getattr(model, attr) for model in self.models]

        # Add kernel attributes
        for attr in prediction_attributes:
            for operation in operations:
                for model in self.models:
                    value = getattr(model.predictions[-1], attr)
                    if value is None:
                        models_dict[f"{operation.__name__}({attr})"].append(None)
                    else:
                        models_dict[f"{operation.__name__}({attr})"].append(
                            operation(value)
                        )

        # Add kernel attributes
        if kernel_attributes:
            d = gpflow.utilities.parameter_dict(self.models[0].model)

            for key in d.keys():
                models_dict[key] = [
                    gpflow.utilities.parameter_dict(model.model)[key].numpy().item()
                    for model in self.models
                ]

        # Set colors and labels for models
        colors = ["black" for _ in self.models]  # default colors
        labels_legend = []
        colors_legend = []

        if colored_models:
            colors = [model.color for model in self.models]
            for model in self.models:
                if model.color not in colors_legend and model.color != "none":
                    colors_legend.append(model.color)
                    labels_legend.append(model.label.replace("_", "-"))

        # Create legend
        legend = []
        for i in range(len(labels_legend)):
            legend.append(
                mpatches.Patch(color=colors_legend[i], label=labels_legend[i])
            )

        # Plot legend separately, as it can get large
        _, ax = plt.subplots(figsize=(len(legend), 0.5))
        ax.axis("off")
        ax.legend(handles=legend, loc="center", ncols=len(legend))
        plt.savefig(save_fig / f"_legend.{file_format}")
        plt.close()

        # Plot data
        keys = list(models_dict.keys())
        for i in range(len(keys) - 1):
            value_1 = keys[i]
            for value_2 in keys[i + 1 :]:
                plt.figure()
                plt.title(
                    rf"{value_1.replace('_', '-')} vs. {value_2.replace('_', '-')}"
                )
                plt.scatter(
                    models_dict[value_1],
                    models_dict[value_2],
                    c=colors,
                    s=5,
                    edgecolors="none",
                    alpha=0.7,
                )
                plt.xlabel(f"{value_1.replace('_', '-')}")
                plt.ylabel(f"{value_2.replace('_', '-')}")
                plt.tight_layout()
                plt.grid()
                plt.savefig(save_fig / f"{value_1}_x_{value_2}.{file_format}")
                plt.close()

    def models_to_dict(self) -> Dict[str, Model]:
        """Put models into a dictionary based on the light curve info.

        Returns:
            Dict[str, Model]: Dictionary with light curve info as a key and model as a value.
        """
        models_dict = {}
        for model in self.models:
            models_dict[model.light_curve.info] = model
        return models_dict
