import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Iterator
import matplotlib.pyplot as plt
from .exoplanet import Exoplanet
import json
from exo_search.utils.logger import Logger, Category
from exo_search.entities.exoplanet import Exoplanet
from pathlib import Path
from exo_search.utils.normalization import z_score
from exo_search.utils.utils import compare_floats


class LightCurve:
    """Represents a single light curve."""

    logger = Logger("light_curve.py")

    # Labels for the x axis used in plots
    xlabels_per_mission = {"TESS": "BTJD [days]", "Kepler": "BKJD [days]"}

    # Batch names for various missions
    batch_names_per_mission = {"TESS": "sector", "Kepler": "quarter"}

    # Time offsets for various missions
    BJD_offset_per_mission = {"TESS": 2457000, "Kepler": 2454833}

    def __init__(
        self,
        star_name: str,
        time: np.ndarray,
        flux: np.ndarray,
        mission: str,
        author: str,
        cadence_period: float,
        batch: List[int],
        exoplanets: List[Exoplanet] = None,
        config_str: str = "",
    ) -> None:
        """Constructor.

        Args:
            star_name (str): name of the star.
            time (np.ndarray): time values.
            flux (np.ndarray): flux values.
            mission (str): mission.
            author (str): author.
            cadence_period (float): cadence period.
            batch (List[int]): list of batches.
            exoplanets (List[Exoplanet], optional): list of Exoplanet objects. Defaults to empty list.
            config_str (str, optional): configuration string. Defaults to "".
        """
        self.star_name = star_name
        self.time = time
        self.flux = flux
        self.mission = mission
        self.author = author
        self.cadence_period = cadence_period
        self.batch = batch
        self.exoplanets: List[Exoplanet] = exoplanets if exoplanets is not None else []
        self.config_str = config_str

    @property
    def batch_name(self) -> str:
        """Batch name usable for the light curve. If unknown, returns "batch".

        Returns:
            str: batch name.
        """
        name = self.batch_names_per_mission.get(self.mission, None)
        if name is None:
            name = "batch"
        return name

    @property
    def xlabel(self) -> str:
        """X-axis label for plots. If unknown, returns "Days".

        Returns:
            str: x-axis label.
        """
        label = self.xlabels_per_mission.get(self.mission, None)
        if label is None:
            label = "Days"
        return label

    @property
    def BJD_offset(self) -> int:
        """BJD offset based on the mission. If unknown, returns 0.

        Returns:
            int: BJD offset in days.
        """
        offset = self.BJD_offset_per_mission.get(self.mission, None)
        if offset is None:
            offset = 0
        return offset

    @property
    def csv_info(self) -> str:
        """Returns the light curve info as a string.

        Returns:
            str: star name, mission, author, cadence period, batches added by +.
        """
        return f"{self.star_name},{self.mission},{self.author},{self.cadence_period},{'+'.join(map(str, self.batch))}"

    @property
    def info(self) -> str:
        """Returns the light curve info as a string.

        Returns:
            str: star name, mission, author, cadence period, batches as batch=[1, 2].
        """
        return f"{self.star_name},{self.mission},{self.author},{self.cadence_period},batch={self.batch}"

    def __str__(self) -> str:
        """String representation of a LightCurve object.

        Returns:
            str: representation of a LightCurve object.
        """
        return f"LightCurve ({self.star_name}, {self.batch_name}: {self.batch})"

    def plot(
        self,
        show_fig: bool = True,
        save_fig: Path = None,
        plot_exoplanets: bool = True,
    ) -> None:
        """Plot the light curve.

        Args:
            show_fig (bool, optional): displays the figure. Defaults to True.
            save_fig (Path, optional): filepath to save the figure. Figure is not saved by default.
            plot_exoplanets (bool, optional): adds vertical lines representing transits to the figure. Defaults to True.
        """
        _, ax = plt.subplots()

        # Plot time and flux values
        ax.scatter(
            self.time,
            self.flux,
            s=0.5,
            color="black",
            label=self.star_name,
            edgecolors="none",
        )

        # Set texts around the plot
        ax.set_title(f"{self.star_name}", pad=15)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("Flux")
        ax.text(
            0.5,
            1.02,
            f"{self.mission}, {self.author}, c.p.: {self.cadence_period}, {self.batch_name}: {', '.join(map(str, self.batch))}",
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=8,
        )

        plt.tight_layout()

        # Plot exoplanet
        if plot_exoplanets:
            self._plot_exoplanets(ax)

        # Save figure
        if save_fig is not None:
            save_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_fig)

        # Show figure
        if show_fig:
            plt.show()

        # Close figure
        plt.close()

    def _plot_exoplanets(
        self, ax: plt.axes, confirmed: List[str] = ["CP", "KP"]
    ) -> None:
        """Adds vertical lines representing transits to provided axis.

        Args:
            ax (plt.axes): axis.
            confirmed (List[str], optional): list of dispositions to draw as solid line. Defaults to ["CP", "KP"]. Other dispositions are drawn as dashed lines.
        """
        if len(self.exoplanets) == 0:
            return

        # To differentiate between exoplanets, create a color map and color each differently
        cm = plt.get_cmap("rainbow")
        cm_range = np.arange(0, 1, 1 / len(self.exoplanets))
        colors = [cm(i) for i in cm_range]

        # Iterate over exoplanets,
        # if it has all the attributes,
        # calculate the starting transit in the batch, and then display the rest
        for index, exoplanet in enumerate(self.exoplanets):
            if exoplanet.period is None or exoplanet.transit_midpoint_BJD is None:
                continue

            # Calculate the transits
            transit_positions = self.get_transits(exoplanet)

            # Create label for the exoplanet lines
            label = f"{exoplanet.name}, {round(exoplanet.period, 2) if exoplanet.period else '-'}, {exoplanet.disposition}"
            first_label = True

            # Plot transits as vertical lines
            for transit_position in transit_positions:
                linestyle = "solid" if exoplanet.disposition in confirmed else "dashed"
                ax.axvline(
                    transit_position,
                    color=colors[index],
                    linewidth=0.5,
                    linestyle=linestyle,
                    zorder=0,
                    alpha=0.8,
                    label=label if first_label else None,
                )
                first_label = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts a LightCurve object to a dictionary.

        Returns:
            Dict[str, Any]: dictionary representation of the object.
        """
        return {
            "star_name": self.star_name,
            "time": self.time.tolist(),
            "flux": self.flux.tolist(),
            "mission": self.mission,
            "author": self.author,
            "cadence_period": self.cadence_period,
            "batch": self.batch,
            "exoplanets": [e.to_dict() for e in self.exoplanets],
            "config_str": self.config_str,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LightCurve":
        """Create a LightCurve object from a dictionary.

        Args:
            d (Dict[str, Any]): dictionary

        Returns:
            LightCurve: newly created LightCurve object.
        """
        return LightCurve(
            d["star_name"],
            np.array(d["time"]),
            np.array(d["flux"]),
            d["mission"],
            d["author"],
            d["cadence_period"],
            d["batch"],
            [Exoplanet.from_dict(e) for e in d["exoplanets"]],
            d["config_str"],
        )

    def __eq__(self, other: "LightCurve") -> bool:
        """Compares LightCurve objects.

        Args:
            other (LightCurve): other LightCurve object.

        Returns:
            bool: True if the light curves have the same attributes.
        """
        if other is None:
            return False
        return (
            self.star_name == other.star_name
            and self.mission == other.mission
            and self.author == other.author
            and compare_floats(self.cadence_period, other.cadence_period)
            and np.allclose(self.time, other.time)
            and np.allclose(self.flux, other.flux)
            and set(self.batch) == set(other.batch)
            and self.config_str == other.config_str
            and self.exoplanets == other.exoplanets
        )

    def matches(
        self,
        star_name: str,
        mission: str,
        author: str,
        cadence_period: float,
        batch: List[int],
    ) -> bool:
        """Checks if the light curve matches provided parameters.

        Args:
            star_name (str): name of the star.
            mission (str): mission.
            author (str): author.
            cadence_period (float): cadence period.
            batch (List[int]): list of batches.

        Returns:
            bool: True if the light curve matches.
        """
        return (
            self.star_name == star_name
            and self.mission == mission
            and self.author == author
            and compare_floats(self.cadence_period, cadence_period)
            and set(self.batch) == set(batch)
        )

    def downsample(
        self,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> "LightCurve":
        """Return new downsampled light curve.

        Args:
            method (Callable): downsampling method.

        Returns:
            LightCurve: downsampled light curve.
        """
        time, flux = method(self.time, self.flux, *args, **kwargs)
        return LightCurve(
            self.star_name,
            time,
            flux,
            self.mission,
            self.author,
            self.cadence_period,
            self.batch,
            self.exoplanets,
            self.config_str,
        )

    def get_normalized_flux(self) -> np.ndarray:
        """Z-score normalizes flux.

        Returns:
            np.ndarray: z-score normalized flux.
        """
        return z_score(self.flux)

    @staticmethod
    def join_lc(
        star_name: str,
        light_curves: List["LightCurve"],
        mission: str,
        author: str,
        cadence_period: float,
        exoplanets: List[Exoplanet] = None,
        config_str: str = "",
    ) -> "LightCurve":
        """Combines multiple light curves into one. New light curve has the provided attributes. Flux is z-score normalized.

        Args:
            star_name (str): name of the star.
            light_curves (List[LightCurve]): light curves to be combined.
            mission (str): mission for the new light curve.
            author (str): author for the new light curve.
            cadence_period (float): cadence period for the new light curve.
            exoplanets (List[Exoplanet], optional): list of exoplanets for the new light curve. Defaults to None.
            config_str (str, optional): configuration string for the new light curve. Defaults to "".

        Returns:
            LightCurve: new LightCurve object.
        """
        # If no light curves are provided, return None
        if len(light_curves) == 0:
            return None

        # Normalize and concatenate flux values across all the light curves
        normalized_flux = np.concatenate(
            [lc.get_normalized_flux() for lc in light_curves]
        )

        # Create new light curve with new flux and time values and list of batches
        return LightCurve(
            star_name=star_name,
            time=np.concatenate([lc.time for lc in light_curves]),
            flux=normalized_flux,
            mission=mission,
            author=author,
            cadence_period=cadence_period,
            batch=[batch for lc in light_curves for batch in lc.batch],
            exoplanets=exoplanets,
            config_str=config_str,
        )

    def get_transits(self, e: Exoplanet) -> List[float]:
        """Calculates transit positions for the provided exoplanet.

        Args:
            e (Exoplanet): exoplanet.

        Returns:
            List[float]: list of transit times.
        """
        if e.period is None or e.transit_midpoint_BJD is None:
            return []

        # Calculate the first transit
        midpoint = e.transit_midpoint_BJD - self.BJD_offset
        first_transit = (
            midpoint + np.ceil((self.time[0] - midpoint) / e.period) * e.period
        )
        # Calculate the rest of the transits
        transits = np.arange(first_transit, self.time[-1], e.period).tolist()

        return transits

    def save_to_file(self, filename: Path) -> None:
        """Save LightCurve object to a JSON file.

        Args:
            filename (Path): JSON filename.
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        except OSError as e:
            self.logger.log(
                Category.ERROR,
                f"LC {self.star_name}",
                "save_to_file()",
                f"failed to save LightCurve ({e})",
            )

    @staticmethod
    def load_from_file(filename: Path) -> "LightCurve":
        """Load LightCurve object from a JSON file.

        Args:
            filename (Path): JSON filename.

        Returns:
            LightCurve: loaded LightCurve, or None.
        """
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except OSError as e:
            LightCurve.logger.log(
                Category.ERROR,
                filename,
                "load_from_file()",
                f"failed to load LightCurve ({e})",
            )
            return None
        except json.JSONDecodeError as e:
            LightCurve.logger.log(
                Category.ERROR, filename, "load_from_file()", f"Bad JSON ({e})"
            )
            return None

        return LightCurve.from_dict(data)

    def fill_gaps(self, gap: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fills gaps in time and flux.

        Args:
            gap (float, optional): expected gap between the data points. Is calculated from the cadence period by default.

        Returns:
            Tuple[np.ndarray, np.ndarray]: time and flux values with filled missing values. Flux values are filled with np.nan.
        """

        # Calculate the gap if not provided
        if gap is None:
            gap = self.cadence_period / (60 * 60 * 24)

        # Calculate the time
        new_time = np.arange(self.time[0], self.time[-1], gap)

        # Add to the new flux either the original values, or np.nan,
        # based on the time values
        new_flux = []
        j = 0
        for i in range(len(new_time)):
            if abs(self.time[j] - new_time[i]) < gap:
                new_flux.append(self.flux[j])
                j += 1
            else:
                new_flux.append(np.nan)

        return (new_time, new_flux)


def load_lcs_from_dir(d: Path, n: int = None) -> List[LightCurve]:
    """Load a list of LightCurve objects from a provided directory.

    Args:
        d (Path): directory.
        n (int, optional): load from n-th JSON file from the directory. All files are loaded by default.

    Returns:
        List[LightCurve]: list of LightCurve objects.
    """

    if not d.exists():
        return []

    # List of files in the directory
    filenames = sorted([item.name for item in d.iterdir() if item.suffix == ".json"])

    lcs = []
    # Load only the n-th file
    if n is not None:
        if n > len(filenames) or n < 1:
            return []
        filenames = [filenames[n - 1]]

    # For each file, load the LightCurve object
    for fn in filenames:
        lc = LightCurve.load_from_file(d / fn)
        if lc is None:
            print(f"Failed to load: {fn}")
            continue
        lcs.append(lc)

    return lcs


def load_lcs_from_dir_list(d: Path, names: List[str]) -> List[LightCurve]:
    """Load LightCurve objects from a directory based on provided names. Names are tested for inclusion, not equality.

    Args:
        d (Path): directory.
        names (List[str]): list of substrings to look for.

    Returns:
        List[LightCurve]: list of LightCurve objects.
    """

    def in_name(names: List[str], s: str) -> bool:
        """Check if the string contains anything from names.

        Args:
            names (List[str]): list of string to check.
            s (str): string to compare the list againts.

        Returns:
            bool: True if s contains any name from names.
        """

        for name in names:
            if name in s:
                return True
        return False

    if not d.exists():
        return []

    # List of filenames
    filenames = [
        item.name
        for item in d.iterdir()
        if item.suffix == ".json" and in_name(names, item.name)
    ]

    # For each file, load the LightCurve object
    lcs = []
    for fn in filenames:
        lc = LightCurve.load_from_file(d / fn)
        if lc is None:
            print(f"Failed to load: {fn}")
            continue
        lcs.append(lc)

    return lcs


def iterate_lcs_from_dir(d: Path) -> Iterator[LightCurve]:
    """Iterate over LightCurve objects from the directory.

    Args:
        d (Path): directory.

    Yields:
        Iterator[LightCurve]: LightCurve objects.
    """
    if not d.exists():
        return

    # List of filenames
    filenames = sorted([item.name for item in d.iterdir() if item.suffix == ".json"])

    # For each file, yield LightCurve object
    for fn in filenames:
        lc = LightCurve.load_from_file(d / fn)
        if lc is None:
            print(f"Failed to load: {fn}")
            continue
        yield lc
