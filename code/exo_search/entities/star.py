from typing import List, Dict, Any, Iterator
import json
from .exoplanet import Exoplanet
from .light_curve import LightCurve
import pandas as pd
import lightkurve as lk
from astropy.table import Table
from exo_search.utils.logger import Logger, Category
import numpy as np
import time
from random import randrange
import io
from pathlib import Path

lk.conf.cache_dir = "./.lk_cache"


class Star:
    """Represents a single star."""

    # How many attempts to perform when downloading list of observations/light curves
    download_attempts = 1

    # How many seconds to sleep before download attempts
    download_sleep = 30

    # Multiplicator for the sleep between each download attempt
    download_multiplicator = 3

    # Value from this range +- is added to the total sleep length
    download_range = 20

    logger = Logger("star.py")

    def __init__(
        self,
        primary_name: str,
        names: List[str] = None,
        exoplanets: List[Exoplanet] = None,
        light_curves: List[LightCurve] = None,
        search_result: pd.DataFrame = None,
    ) -> None:
        """Constructor

        Args:
            primary_name (str): name of the star.
            names (List[str], optional): list of all the names of the star. Defaults to a list containing the primary name.
            exoplanets (List[Exoplanet], optional): list of Exoplanet objects. Defaults to empty list.
            light_curves (List[LightCurve], optional): list of LightCurve objects. Defaults to empty list.
            search_result (pd.DataFrame, optional): pandas dataframe with available observations. Defaults to None.
        """
        self.primary_name = primary_name
        self.names = names if names is not None else [primary_name]
        self.exoplanets = exoplanets if exoplanets is not None else []
        self.light_curves = light_curves if light_curves is not None else []
        self.search_result = search_result

    def __str__(self) -> str:
        """String representation of a Star object.

        Returns:
            str: representation of a Star object.
        """
        return f"Star ({self.primary_name})"

    def __eq__(self, other: "Star") -> bool:
        """Compares Star objects.

        Args:
            other (Star): other Star object.

        Returns:
            bool: True if the stars have the same attributes.
        """
        return (
            self.primary_name == other.primary_name
            and set(self.names) == set(other.names)
            and self.exoplanets == other.exoplanets
            and self.light_curves == other.light_curves
            and self.search_result.equals(other.search_result)
            if self.search_result is not None or other.search_result is not None
            else self.search_result == other.search_result
        )

    def get_lc(
        self,
        mission: str,
        author: str,
        cadence_period: float,
        n: int = None,
        separate: bool = False,
    ) -> List[LightCurve]:
        """Returns a list of LightCurve objects matching the provided parameters.

        Args:
            mission (str): mission.
            author (str): author.
            cadence_period (float): cadence period.
            n (int, optional): number of LightCurve objects to return. Returns all by default. If invalid, results all by default.
            separate (bool, optional): if True, LightCurve objects are returned separately. If False, matching LightCurve objects are joined and returned as one. Defaults to False.

        Returns:
            List[LightCurve]: list of LightCurve objects.
        """

        # Get light curves that match the parameters
        light_curves = self.get_filtered_lc(mission, author, cadence_period)

        # Work with only the first n
        if n is not None:
            if 0 < n <= len(light_curves):
                light_curves = light_curves[:n]

        # List of light curves
        lcs = []
        if not separate:
            # Join light curves into one

            # List of exoplanets
            exoplanets = [exo for l in light_curves for exo in l.exoplanets]
            # Create new config string if there are more than 1 light curve, otherwise keep the original
            config_str = ""
            if len(light_curves) == 1:
                config_str = light_curves[0].config_str
            else:
                batches = [b for lc in light_curves for b in lc.batch]
                config_str = f"{self.primary_name},{mission},{author},{cadence_period},batch={batches}"

            # Join the light curves
            lc = LightCurve.join_lc(
                self.primary_name,
                light_curves,
                mission,
                author,
                cadence_period,
                exoplanets,
                config_str=config_str,
            )

            if lc is not None:
                lcs.append(lc)
        else:
            # If to separate, return the LightCurves separately, but with normalized flux.
            for lc in light_curves:
                lc.flux = lc.get_normalized_flux()
                lcs.append(lc)

        return lcs

    def download_lc(
        self, mission: str, author: str, cadence_period: float, n: int = 1
    ) -> bool:
        """Download light curves.

        Args:
            mission (str): mission.
            author (str): author
            cadence_period (float): cadence period.
            n (int, optional): how many batches to download. If None, downloads all. Defaults to 1.

        Returns:
            bool: True on success.
        """

        # Get the search result (list of available observations)
        search_result = self.get_search_result()
        if not self.search_result_available():
            self.logger.log(
                Category.ERROR,
                self.primary_name,
                "download_lc()",
                "SKIPPING: failed to download search result",
            )
            return False

        # Filter search result by specified parameters
        search_result = search_result[search_result["project"] == mission]
        search_result = search_result[search_result["author"] == author]
        search_result = search_result[search_result["exptime"] == cadence_period]

        # Sort search result by batch number
        search_result = search_result.sort_values(by="sequence_number")

        # If the search result is empty, log error and exit wth error
        if len(search_result) == 0:
            self.logger.log(
                Category.ERROR,
                self.primary_name,
                "download_lc_first_n()",
                "no light curves available for download",
            )
            return False

        # Download only the first n,
        # if incorrect, download all
        if n is not None:
            if 0 < n <= len(search_result):
                search_result = search_result.iloc[:n]
            else:
                self.logger.log(
                    Category.WARNING,
                    self.primary_name,
                    "download_lc_first_n()",
                    f"bad n ({n}); number of available light curves ({len(search_result)})",
                )

        # Create a list of light curves to download;
        # iterate over the search result and compare each light curve with already downloaded light curves
        lc_to_download = []
        for _, lc in search_result.iterrows():
            # Based on the mission, extract the batch number
            batch = None
            if lc["project"] == "TESS":
                batch = [lc["sequence_number"]]
            elif lc["project"] == "Kepler":
                batch = [int(lc["mission"][len("Kepler Quarter ") :])]

            # Compare with all downloaded light curves
            # if no such light curve is downloaded, add it to the list for download
            if not self.check_is_downloaded(
                lc["project"], lc["author"], lc["exptime"], batch
            ):
                lc_to_download.append(lc)

        # If there are no light curves to be downloaded, log info end exit with success
        if len(lc_to_download) == 0:
            self.logger.log(
                Category.INFO,
                self.primary_name,
                "download_lc_first_n()",
                f"downloading 0 light curves (all found locally)",
            )
            return True

        # Convert list of light curves to be downloaded to a lightkurve SearchResult
        lc_to_download = lk.SearchResult(
            Table.from_pandas(pd.DataFrame(lc_to_download))
        )

        # Try to download light curves;
        # if it fails, sleep and try again
        lc_downloaded = None
        for i in range(self.download_attempts):
            self.logger.log(
                Category.INFO,
                self.primary_name,
                "download_lc_first_n()",
                f"downloading {len(lc_to_download)} light curves, attempt {i+1}/{self.download_attempts}",
            )

            try:
                lc_downloaded = lc_to_download.download_all()
            except Exception as e:
                # If the last attempt failed, log error
                # otherwise log error and sleep
                if i == self.download_attempts - 1:
                    self.logger.log(
                        Category.ERROR,
                        self.primary_name,
                        "download_lc()",
                        f"failed to download light curves ({e})",
                    )
                else:
                    sleep = self.download_sleep * pow(self.download_multiplicator, i)
                    sleep += randrange((-1) * self.download_range, self.download_range)
                    sleep = max(0, sleep)
                    self.logger.log(
                        Category.ERROR,
                        self.primary_name,
                        "download_lc()",
                        f"failed to download light curves ({e}), going to sleep for {sleep}s",
                    )
                    time.sleep(sleep)
                continue

            # If something got downloaded, break the loop
            if lc_downloaded != None:
                break

        # If nothing or bad data got downloaded, log and exit with error
        if not lc_downloaded:
            self.logger.log(
                Category.ERROR,
                self.primary_name,
                "download_lc()",
                "SKIPPING: failed to download light curves",
            )
            return False

        # Otherwise log success
        self.logger.log(
            Category.SUCCESS,
            self.primary_name,
            "download_lc_first_n()",
            "downloaded OK",
        )

        # Convert downloaded light curves to LightCurve objects
        lc_objects = []
        for lc in lc_downloaded:
            batch = None
            if lc.mission == "TESS":
                batch = [lc.sector]
            elif lc.mission == "Kepler":
                batch = [lc.quarter]

            df = lc.to_pandas().reset_index()
            df = df[["time", "pdcsap_flux"]]
            df = df.dropna()

            lc_objects.append(
                LightCurve(
                    self.primary_name,
                    df["time"].to_numpy(),
                    df["pdcsap_flux"].to_numpy(),
                    mission,
                    author,
                    cadence_period,
                    batch,
                )
            )

        for lc in lc_objects:
            self.light_curves.append(lc)

        return True

    def get_search_result(self) -> pd.DataFrame:
        """Get search result from lk.search_lightcurve() for the star. If search result is missing, it is downloaded and set.

        Returns:
            pd.DataFrame: search result.
        """
        if self.search_result_available():
            return self.search_result

        # The search result is not available, download it;
        # iteratively try, if it fails, sleep and try again
        for i in range(self.download_attempts):
            # Try for all the names of the star
            for name in self.names:
                self.logger.log(
                    Category.INFO,
                    self.primary_name,
                    "get_search_result()",
                    f"fetching search result for {name}, attempt {i+1}/{self.download_attempts}",
                )
                try:
                    search_result = lk.search_lightcurve(name)
                except Exception as e:
                    # if error occurred during download, log error and try for different name
                    self.logger.log(
                        Category.ERROR,
                        self.primary_name,
                        "get_search_result()",
                        f"fetch search result for {name} crashed ({e})",
                    )
                    continue

                # If success, convert to pandas
                search_result = search_result.table.to_pandas()

                # If it is empty, log error and try for different name
                if search_result.empty:
                    self.logger.log(
                        Category.ERROR,
                        self.primary_name,
                        "get_search_result()",
                        f"unable to fetch search result for {name}",
                    )
                    continue

                # Otherwise log success, set the search result and return it
                self.logger.log(
                    Category.SUCCESS,
                    self.primary_name,
                    "get_search_result()",
                    "fetched OK",
                )
                self.search_result = search_result
                return search_result

            # If it is not the last try, log error, sleep and try again
            if i < self.download_attempts - 1:
                sleep = self.download_sleep * pow(self.download_multiplicator, i)
                sleep += randrange((-1) * self.download_range, self.download_range)
                sleep = max(0, sleep)
                self.logger.log(
                    Category.ERROR,
                    self.primary_name,
                    "get_search_result()",
                    f"failed to fetch search result for ANY name of {self.primary_name}: {self.names}, going to sleep for {sleep}s",
                )
                time.sleep(sleep)

        # otherwise log error and return None
        self.logger.log(
            Category.ERROR,
            self.primary_name,
            "get_search_result()",
            f"failed to fetch search result for ANY name of {self.primary_name}: {self.names}",
        )
        return None

    def save_to_file(self, filename: Path) -> None:
        """Save the Star object to JSON file.

        Args:
            filename (str, optional): JSON filename.
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        except OSError as e:
            self.logger.log(
                Category.ERROR,
                self.primary_name,
                "save_to_file()",
                f"Failed to save star to file ({e}).",
            )
            return

        self.logger.log(
            Category.SUCCESS,
            self.primary_name,
            "save_to_file()",
            "Star saved successfully.",
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Star":
        """Create a Star object from a dictionary.

        Args:
            d (Dict[str, Any]): dictionary.

        Returns:
            Star: newly created Star object.
        """

        # Deserialize light curves
        light_curves = [LightCurve.from_dict(lc) for lc in d["light_curves"]]
        # Deserialize exoplanets
        exoplanets = [Exoplanet.from_dict(exoplanet) for exoplanet in d["exoplanets"]]
        # Deserialize search result
        search_result = None

        if d["search_result"]:
            search_result = pd.read_json(io.StringIO(d["search_result"]))

        return Star(
            primary_name=d["primary_name"],
            names=d["names"],
            exoplanets=exoplanets,
            light_curves=light_curves,
            search_result=search_result,
        )

    @staticmethod
    def load_from_file(filename: Path) -> "Star":
        """Load Star objects from JSON file.

        Args:
            filename (Path): JSON filename.

        Returns:
            Star: newly creates Star object.
        """

        try:
            with open(filename) as f:
                d = json.load(f)
        except OSError as e:
            Star.logger.log(
                Category.ERROR,
                filename,
                "load_from_file()",
                f"Failed to load star from file ({e}).",
            )
            return None
        except json.decoder.JSONDecodeError as e:
            Star.logger.log(
                Category.ERROR, filename, "load_from_file()", f"Bad JSON. ({e})"
            )
            return None

        Star.logger.log(
            Category.SUCCESS,
            d["primary_name"],
            "load_from_file()",
            "Star loaded successfully.",
        )
        return Star.from_dict(d)

    def check_is_downloaded(
        self, mission: str, author: str, cadence_period: float, batch: List[int]
    ) -> bool:
        """Based on arguments, check whether the light curves is already downloaded.

        Args:
            mission (str): mission.
            author (str): author.
            cadence (float): cadence period.
            batch (List[int]): batch numbers.

        Returns:
            bool: True if light curve is downloaded
        """
        for lc in self.light_curves:
            if lc.matches(
                star_name=self.primary_name,
                mission=mission,
                author=author,
                cadence_period=cadence_period,
                batch=batch,
            ):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert Star object to a dictionary.

        Returns:
            Dict[str, Any]: serialized Star object as dictionary.
        """
        return {
            "primary_name": self.primary_name,
            "names": self.names,
            "exoplanets": [e.to_dict() for e in self.exoplanets],
            "light_curves": [lc.to_dict() for lc in self.light_curves],
            "search_result": (
                self.search_result.to_json(orient="records")
                if self.search_result_available()
                else None
            ),
        }

    def get_filtered_lc(
        self,
        mission: str = None,
        author: str = None,
        cadence_period: float = None,
    ) -> List[LightCurve]:
        """Filter light curves based on provided parameters

        Args:
            mission (str, optional): mission. Not matched by default.
            author (str, optional): author. Not matched by default.
            cadence_period (float, optional): cadence period.Not matched by default.

        Returns:
            List[LightCurve]: list of light curves.
        """

        # Filter light curves based on provided parameters
        light_curves = []
        for lc in self.light_curves:
            if mission is not None and mission != lc.mission:
                continue
            if author is not None and author != lc.author:
                continue
            if cadence_period is not None and cadence_period != lc.cadence_period:
                continue
            light_curves.append(lc)

        # Sort by the batch;
        # if multiple batches are available, sort the batches and take the first one
        light_curves.sort(key=lambda lc: sorted(lc.batch)[0])

        return light_curves

    def search_result_available(self) -> bool:
        """Check whether search result is available.

        Returns:
            bool: True if search_result is available.
        """
        if type(self.search_result) == pd.DataFrame:
            return True
        return False


def load_stars_from_dir(d: Path, n: int = None) -> List[Star]:
    """Load star objects from directory.

    Args:
        d (Path): directory.
        n (int, optional): load only the n-th JSON file. All JSON files are loaded by default.

    Returns:
        List[Star]: list of stars.
    """
    if not d.exists():
        return []

    # List of filenames
    filenames = sorted([item.name for item in d.iterdir() if item.suffix == ".json"])

    # Load only the n-th file
    # If incorrect, load all files
    if n is not None:
        if n > len(filenames) or n < 1:
            return []

        filenames = [filenames[n - 1]]

    stars = []
    # Iterate over filenames and load the Star objects
    for fn in filenames:
        star = Star.load_from_file(d / fn)
        if star is None:
            print(f"Failed to load: {fn}")
            continue
        stars.append(star)

    return stars


def iterate_stars_from_dir(d: Path, start: int = None, end: int = None) -> Iterator[Star]:
    """Iterate over Star objects in a directory.

    Args:
        d (Path): directory.
        start (int, optional): start index. Defaults to the first.
        end (int, optional): end index. Defaults to the last.

    Yields:
        Iterator[Star]: yields Star objects.
    """

    if not d.exists():
        return

    # List of filenames
    filenames = sorted([item.name for item in d.iterdir() if item.suffix == ".json"])

    # Filter the list by start and end
    if start is not None and end is not None:
        filenames = filenames[start:end]
    elif start is not None:
        filenames = filenames[start:]
    elif end is not None:
        filenames = filenames[:end]

    # Iterate over the filenames and yield the loaded Star object
    for fn in filenames:
        star = Star.load_from_file(d / fn)
        if star is None:
            print(f"Failed to load: {fn}")
            continue

        yield star
