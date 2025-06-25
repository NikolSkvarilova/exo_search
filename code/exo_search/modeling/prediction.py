import numpy as np
from typing import Dict, Any


class Prediction:
    """Represents a single prediction."""

    def __init__(
        self,
        x: np.ndarray,
        f_mean: np.ndarray = None,
        y_mean: np.ndarray = None,
        f_var: np.ndarray = None,
        y_var: np.ndarray = None,
    ) -> None:
        """Constructor.

        Args:
            x (np.ndarray): x-values.
            f_mean (np.ndarray, optional): predicted f mean. Defaults to None.
            y_mean (np.ndarray, optional): predicted y mean. Defaults to None.
            f_var (np.ndarray, optional): predicted f variance. Defaults to None.
            y_var (np.ndarray, optional): predicted y variance. Defaults to None.
        """

        self.x = x
        self.f_mean = f_mean
        self.y_mean = y_mean
        self.f_var = f_var
        self.y_var = y_var
        self.f_2s = (
            2 * np.sqrt(f_var)
            if f_var is not None and not np.isnan(np.min(f_var))
            else None
        )
        self.y_2s = (
            2 * np.sqrt(y_var)
            if y_var is not None and not np.isnan(np.min(y_var))
            else None
        )

    @property
    def plot_x(self) -> np.ndarray:
        """1D x-values for plots.

        Returns:
            np.ndarray: x-vales.
        """
        return self.x[:, 0]

    def predict_f(self, model: "gpflow.models.GPR") -> None:
        """Predict the f values.

        Args:
            model (gpflow.models.GPR): GPR model.
        """
        self.f_mean, self.f_var = list(
            map(lambda x: x.numpy(), model.predict_f(self.x, full_cov=False))
        )
        self.f_2s = 2 * np.sqrt(self.f_var)

    def predict_y(self, model: "gpflow.models.GPR") -> None:
        """Predict the Y values.

        Args:
            model (gpflow.models.GPR): GPR model.
        """
        self.y_mean, self.y_var = list(
            map(lambda x: x.numpy(), model.predict_y(self.x))
        )
        self.y_2s = 2 * np.sqrt(self.y_var)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Prediction object to a dictionary.

        Returns:
            Dict[str, Any]: Prediction object as a dictionary.
        """
        return {
            "x": self.x.tolist(),
            "f_mean": self.f_mean.tolist() if self.f_mean is not None else None,
            "y_mean": self.y_mean.tolist() if self.y_mean is not None else None,
            "y_var": self.y_var.tolist() if self.y_var is not None else None,
            "f_var": self.f_var.tolist() if self.f_var is not None else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Prediction":
        """Creates Prediction object from a dictionary.

        Args:
            d (Dict[str, Any]): dictionary representation of the oject.

        Returns:
            Prediction: Prediction object.
        """
        return Prediction(
            np.array(d["x"]),
            np.array(d["f_mean"]) if d["f_mean"] is not None else None,
            np.array(d["y_mean"]) if d["y_mean"] is not None else None,
            np.array(d["f_var"]) if d["f_var"] is not None else None,
            np.array(d["y_var"]) if d["y_var"] is not None else None,
        )

    def matches(self, x: np.ndarray) -> bool:
        """Check whether the prediction matches the arguments.

        Args:
            x (np.ndarray): the x values for the prediction.

        Returns:
            bool: True if matches.
        """
        return np.array_equal(self.x, x)

    def plot(self, ax: "matplotlib.pyplot.axis", is_transit: bool = False) -> None:
        """Plot the f and Y prediction, if available.

        Args:
            ax (matplotlib.pyplot.axis): axis for the plot.
            in_transit (bool): if True, the x axis is in hours.
        """

        x_plot = self.plot_x
        # If the prediction is of a transit model, the x axis is in hours
        if is_transit:
            x_plot *= 24

        # Plot the f-values
        if self.f_mean is not None:
            ax.plot(
                x_plot,
                self.f_mean[:, 0],
                label="Mean",
                color="blue",
                linewidth=0.5,
            )
            if self.f_2s is not None:
                # Confidence interval
                ax.fill_between(
                    x_plot,
                    (self.f_mean - self.f_2s)[:, 0],
                    (self.f_mean + self.f_2s)[:, 0],
                    color="blue",
                    alpha=0.3,
                    label="f 95% CI",
                    edgecolors="none",
                )

        # Plot the y-values
        if self.y_mean is not None and self.y_2s is not None:
            # Confidence interval
            ax.fill_between(
                x_plot,
                (self.y_mean - self.y_2s)[:, 0],
                (self.y_mean + self.y_2s)[:, 0],
                color="tab:red",
                alpha=0.3,
                label="Y 95% CI",
                edgecolors="none",
            )
