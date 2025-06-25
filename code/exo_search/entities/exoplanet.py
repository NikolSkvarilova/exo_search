from typing import Dict, Any
import pandas as pd
from exo_search.utils.utils import compare_floats


class Exoplanet:
    """Represents a single exoplanet."""

    def __init__(
        self,
        name: str,
        period: float = None,
        disposition: str = None,
        transit_midpoint_BJD: float = None,
        transit_duration_h: float = None,
    ) -> None:
        """Constructor.

        Args:
            name (str): name of the exoplanet.
            period (float, optional): orbit period. Defaults to None.
            disposition (str, optional): e.g. confirmed, false positive. Defaults to None.
            transit_midpoint_BJD (float, optional): time value of a middle of a transit. Defaults to None.
            transit_duration_h (float, optional): duration of the transit in hours. Defaults to None.
        """
        self.name = name
        self.period = period
        self.disposition = disposition
        self.transit_midpoint_BJD = transit_midpoint_BJD
        self.transit_duration_h = transit_duration_h

    def __str__(self):
        """String representation of an Exoplanet object.

        Returns:
            str: representation of an Exoplanet object.
        """
        return f"Exoplanet {self.name} [{self.disposition}, {self.period}]"

    def __eq__(self, other: "Exoplanet") -> bool:
        """Compares Exoplanet objects.

        Args:
            other (Exoplanet): other Exoplanet object.

        Returns:
            bool: True if the exoplanets have the same attributes.
        """
        return (
            self.name == other.name
            and compare_floats(self.period, other.period)
            and self.disposition == other.disposition
            and compare_floats(self.transit_midpoint_BJD, other.transit_midpoint_BJD)
            and compare_floats(self.transit_duration_h, other.transit_duration_h)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Exoplanet object to a dictionary.

        Returns:
            Dict[str, Any]: dictionary representation of the object.
        """
        return self.__dict__

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Exoplanet":
        """Creates an Exoplanet object from a dictionary.

        Args:
            d (Dict[str, Any]): dictionary.

        Returns:
            Exoplanet: newly created Exoplanet object.
        """
        return Exoplanet(
            name=d["name"],
            period=d["period"] if not pd.isna(d["period"]) else None,
            disposition=d["disposition"],
            transit_midpoint_BJD=(
                d["transit_midpoint_BJD"]
                if not pd.isna(d["transit_midpoint_BJD"])
                else None
            ),
            transit_duration_h=(
                d["transit_duration_h"]
                if not pd.isna(d["transit_duration_h"])
                else None
            ),
        )

    @staticmethod
    def from_pd_series(series: pd.Series) -> "Exoplanet":
        """Creates an Exoplanet object from a pd.Series representation.

        Args:
            series (pd.Series): pandas Series.

        Returns:
            Exoplanet: newly created Exoplanet object.
        """
        return Exoplanet(
            name=series["name"],
            period=(series["period"] if not pd.isna(series["period"]) else None),
            disposition=series["disposition"],
            transit_midpoint_BJD=(
                series["transit_midpoint_BJD"]
                if not pd.isna(series["transit_midpoint_BJD"])
                else None
            ),
            transit_duration_h=(
                series["transit_duration_h"]
                if not pd.isna(series["transit_duration_h"])
                else None
            ),
        )
