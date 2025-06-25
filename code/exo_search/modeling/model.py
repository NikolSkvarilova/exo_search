import json
from exo_search.entities.light_curve import LightCurve
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from exo_search.utils.logger import Logger, Category
import gpflow
import time
from .prediction import Prediction
from pathlib import Path
import cloudpickle


class Model:
    """Represents a single model."""

    logger = Logger("model.py")

    # Iterations for the training
    training_iterations = 10000

    def __init__(
        self,
        light_curve: LightCurve,
        config_str: str,
        training_time: float = None,
        period: float = None,
        trained: bool = False,
        likelihood: float = None,
        likelihood_var: float = None,
        kernel: gpflow.kernels.Kernel = None,
    ) -> None:
        """Constructor.

        Args:
            light_curve (LightCurve): light curve.
            config_str (str): configuration string used as filename for saving.
            training_time (float, optional): training time. Defaults to None.
            period (float, optional): found period. Defaults to None.
            trained (bool, optional): True if model is already trained. Defaults to False.
            likelihood (float, optional): likelihood. Defaults to None.
            likelihood_var (float, optional): likelihood variance. Defaults to None.
            kernel (gpflow.kernels.Kernel, optional): kernel. Defaults to None. If provided, model is automatically created.
        """

        self.light_curve: LightCurve = light_curve

        self.x = self.light_curve.time[:, None]
        self.y = self.light_curve.flux[:, None]
        self.y = self.y.astype(self.x.dtype)

        self.config_str = config_str

        self.predictions: list[Prediction] = []
        self.training_time = training_time
        self.period = period
        self.trained = trained
        self.likelihood = likelihood
        self.likelihood_var = likelihood_var

        self.model = None

        if kernel is not None:
            self._create_model(kernel)

    def __str__(self) -> str:
        """String representation of a Model object.

        Returns:
            str: representation of a Model object.
        """
        return self.config_str

    @property
    def predictions_count(self) -> int:
        """Number of loaded predictions.

        Returns:
            int: number of loaded predictions.
        """
        return len(self.predictions)

    def save_to_file(self, fn_data: Path, fn_model: Path) -> None:
        """Save model to file

        Args:
            fn_data (str): JSON filename for model data.
            fn_model (str): pickle filename for kernel model parameters.
        """

        fn_data.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        data = self.to_dict()
        try:
            with open(f"{fn_data}", "w") as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "save_to_file()",
                f"Failed to save model to file ({e}).",
            )

        # Save the gpflow model
        fn_model.parent.mkdir(parents=True, exist_ok=True)
        # The parameters are stored as well as the whole kernel
        data = gpflow.utilities.parameter_dict(self.model)
        kernel = self.model.kernel
        try:
            with open(f"{fn_model}", "wb") as f:
                cloudpickle.dump(
                    {"params": data, "kernel": kernel},
                    f,
                )
        except OSError as e:
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "save_model()",
                f"Failed to save gpflow model to file ({e}).",
            )
            return

        self.logger.log(
            Category.SUCCESS,
            self.config_str,
            "save_model()",
            f"Model saved to {fn_data} and {fn_model}.",
        )

    @staticmethod
    def load_from_file(fn_data: Path, fn_model: Path) -> "Model":
        """Load Model object from a JSON file and a pickle file.

        Args:
            fn_data (Path): JSON filepath. Stores the Model object,
            fn_model (Path): pickle filepath. Stores the gpflow model.

        Returns:
            Model: loaded Model object.
        """
        # Load the Model object
        try:
            with open(fn_data, "r") as f:
                d = json.load(f)
        except OSError as e:
            Model.logger.log(
                Category.ERROR, fn_data, "load_from_file()", f"Model not found ({e})"
            )
            return None
        except json.JSONDecodeError as e:
            Model.logger.log(
                Category.ERROR, fn_data, "load_from_file()", f"Bad JSON ({e})"
            )
            return None

        # Deserialize light curve
        lc = LightCurve.from_dict(d["light_curve"])

        # Load the gpflow model
        try:
            with open(fn_model, "rb") as f:
                data = cloudpickle.load(f)
        except OSError as e:
            Model.logger.log(
                Category.ERROR,
                fn_model,
                "load_model()",
                f"Failed to load gpflow model from file ({e})",
            )
            return None

        # Deserialize the gpflow model
        d["kernel"] = data["kernel"]
        d["light_curve"] = lc
        model = Model.from_dict(d)
        try:
            gpflow.utilities.multiple_assign(model.model, data["params"])
        except Exception as e:
            print(f"Failed to load model {e}")
            Model.logger.log(
                Category.ERROR,
                fn_model,
                "load_model()",
                f"Failed to build gpflow model ({e})",
            )

        return model

    def _create_model(self, kernel: gpflow.kernels.Kernel) -> None:
        """Create gpflow model.

        Args:
            kernel (gpflow.kernels.Kernel): kernel.
        """
        if kernel is None or len(self.x) != len(self.y):
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "create_model()",
                f"bad kernel or x, y values",
            )
            self.model = None
            return
        try:
            self.model = gpflow.models.GPR((self.x, self.y), kernel=kernel)
        except Exception as e:
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "create_model()",
                f"creating model failed with: {e}",
            )

    def train(self) -> None:
        """Train model."""
        # If model is already trained, log it and exit
        if self.trained:
            self.logger.log(
                Category.INFO,
                self.config_str,
                "train()",
                "model already trained, skipping training",
            )
            return

        self.logger.log(Category.INFO, self.config_str, "train()", "starting training")

        # Measure time
        start_time = time.time()

        # Training
        opt = gpflow.optimizers.Scipy()
        try:
            opt.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options=dict(maxiter=Model.training_iterations),
            )
        except Exception as e:
            self.logger.log(
                Category.ERROR, self.config_str, "train()", f"training failed with: {e}"
            )
            return

        # Stop time
        self.training_time = time.time() - start_time

        # Calculate likelihood and likelihood var
        # if it fails, the training most likely failed
        try:
            self.likelihood = float(self.model.log_marginal_likelihood().numpy())
            self.likelihood_var = float(self.model.likelihood.variance.numpy())
        except Exception as e:
            self.trained = False
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "train()",
                f"model is probably bad - calculating likelihood or likelihood variance failed with: {e}",
            )
        else:
            self.trained = True
            self.logger.log(
                Category.SUCCESS,
                self.config_str,
                "train()",
                f"training finished, training time: {round(self.training_time, 2)} s, likelihood: {self.likelihood}",
            )

            self.extract_period()

    def predict(self, x: np.ndarray = None, predict_y: bool = False) -> None:
        """Make prediction. If prediction over the same x is already made, use that one.

        Args:
            x (np.ndarray, optional): x values to use for the prediction. Defaults to the model's light curve's time.
            predict_y (bool, optional): if True, besides f-values, the y-values are predicted as well. Defaults to False.
        """

        if x is None:
            x = self.light_curve.time[:, None]

        # Look for already existing prediction
        prediction = None
        for p in self.predictions:
            if p.matches(x):
                prediction = p
                break

        # If none found, create new prediction
        if prediction is None:
            prediction = Prediction(x)
            self.predictions.append(prediction)

        self.logger.log(Category.INFO, self.config_str, "predict()", "predicting")

        # Predict the f values if they do not already exist
        if prediction.f_mean is None:
            prediction.predict_f(self.model)

        # Predict the y values if they do not already exist
        if predict_y and prediction.y_mean is None:
            prediction.predict_y(self.model)

        self.logger.log(
            Category.SUCCESS, self.config_str, "predict()", "prediction done"
        )

    def extract_period(self) -> None:
        """Set period from the model."""
        for key, value in gpflow.utilities.parameter_dict(self.model).items():
            if "period" in key:
                self.period = value.numpy().item()
                break

    def plot(
        self,
        prediction: Prediction = None,
        show_fig: bool = True,
        save_fig: Path = None,
        plot_exoplanets: bool = True,
        extra_title: str = "",
        extra_padding: float = 0,
        is_transit: bool = False,
    ) -> None:
        """Plot prediction.

        Args:
            prediction (Prediction, optional): prediction to plot. Defaults to the latest prediction.
            show_fig (bool, optional): display the figure. Defaults to True.
            save_fig (Path, optional): filename for the figure. Defaults to None.
            plot_exoplanets (bool, optional): if True, transits are marked in the figure. Defaults to True.
            extra_title (str, optional): can contain extra information to be shown above the plot. Defaults to "".
            extra_padding (float, optional): used when extra_title is provided. Defaults to 0.
            is_transit (bool, optional): if True, the x-axis is in hours and transits are not plotted. Defaults to False.
        """

        # Select the prediction
        if prediction is None:
            if len(self.predictions) == 0:
                self.logger.log(
                    Category.ERROR,
                    self.config_str,
                    "plot()",
                    "Failed to plot prediction: no predictions available.",
                )

            prediction = self.predictions[-1]

        # Create figure
        _, ax = plt.subplots()

        # Plot original data
        ax.scatter(
            self.light_curve.time if not is_transit else self.light_curve.time * 24,
            self.light_curve.flux,
            s=0.5,
            color="black",
            edgecolors="none",
        )

        # Plot predictions
        prediction.plot(ax, is_transit)

        # Create title
        title = self.light_curve.star_name

        if is_transit:
            title += f" ({self.light_curve.exoplanets[0].name})"

        ax.set_title(title, pad=15 + extra_padding)

        extra_title = (
            f"{self.light_curve.mission}, {self.light_curve.author}, c.p.: {self.light_curve.cadence_period}, {self.light_curve.batch_name}: {', '.join(map(str, self.light_curve.batch))}"
            + extra_title
        )

        ax.text(
            0.5,
            1.02,
            extra_title,
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=8,
        )

        # Plot transits
        if plot_exoplanets and not is_transit:
            self.light_curve._plot_exoplanets(ax=ax)

        # Set axis labels
        if is_transit:
            ax.set_xlabel("Hours from transit midpoint")
        else:
            ax.set_xlabel(self.light_curve.xlabel)

        ax.set_ylabel("Flux")

        plt.tight_layout()

        # Save fig
        if save_fig is not None:
            save_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_fig)

        # Show fig
        if show_fig:
            plt.show()

        # Close fig
        plt.close()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Model object to a dictionary.

        Returns:
            Dict[str, Any]: dictionary representation of the object.
        """
        return {
            "light_curve": self.light_curve.to_dict(),
            "training_time": self.training_time,
            "period": self.period,
            "config_str": self.config_str,
            "trained": self.trained,
            "likelihood": self.likelihood,
            "likelihood_var": self.likelihood_var,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Model":
        """Create Model object from a dictionary.

        Args:
            d (Dict[str, Any]): dictionary.

        Returns:
            Model: newly created Model object.
        """
        return Model(
            light_curve=d["light_curve"],
            config_str=d["config_str"],
            training_time=d["training_time"],
            period=d["period"],
            trained=d["trained"],
            likelihood=d["likelihood"],
            likelihood_var=(
                d["likelihood_var"] if "likelihood_var" in d.keys() else None
            ),
            kernel=d["kernel"],
        )

    def save_predictions(self, filename: Path) -> None:
        """Save the model's predictions.

        Args:
            filename (Path): directory.
        """

        filename.parent.mkdir(parents=True, exist_ok=True)

        # Save all predictions into a single file
        data = [prediction.to_dict() for prediction in self.predictions]

        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "save_predictions()",
                f"failed to save predictions ({e})",
            )

    def load_predictions(self, filename: Path) -> None:
        """Load prediction.

        Args:
            filename (Path): filename.
        """
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except OSError as e:
            self.logger.log(
                Category.ERROR,
                self.config_str,
                "load_predictions()",
                f"failed to load predictions ({e})",
            )
        except json.JSONDecodeError as e:
            Model.logger.log(
                Category.ERROR, filename, "load_predictions()", f"Bad JSON ({e})"
            )
        else:
            # Predictions successfully loaded,
            # convert them to Prediction objects
            for p in data:
                prediction = Prediction.from_dict(p)
                if prediction is not None:
                    self.predictions.append(prediction)
