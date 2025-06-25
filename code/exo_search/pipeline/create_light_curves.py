from exo_search.entities.star import iterate_stars_from_dir
from pathlib import Path
import pandas as pd
from typing import List


def main(
    stars_path: Path,
    dst_path: Path,
    mission: str,
    author: str,
    cadence_period: float,
    n_batches: int = None,
    separate: bool = False,
    ignore_csv: List[Path] = None,
) -> None:
    """Creates LightCurve objects from Star objects,

    Args:
        stars_path (Path): directory with Star objects.
        dst_path (Path): directory for the new light curves.
        mission (str): mission.
        author (str): author.
        cadence_period (float): cadence period.
        n_batches (int, optional): how many batches to use. All are used by default.
        separate (bool, optional): instead of a single joined light curve, create light curve for each batch. Defaults to False.
        ignore_csv (List[Path], optional): list of csv files with star_names to ignore. Defaults to None.
    """

    # Stars to ignore
    ignore = set()
    if ignore_csv is not None:
        for p in ignore_csv:
            df = pd.read_csv(p)
            ignore.update(df["star_name"])

    dst_path.mkdir(parents=True, exist_ok=True)

    # Iterate over stars
    for star in iterate_stars_from_dir(stars_path):
        # Ignore the star if in the ignore set
        if star.primary_name in ignore:
            continue

        # Get light curves
        lcs = []
        if separate:
            lcs = star.get_lc(mission, author, cadence_period, n_batches, separate=True)
        else:
            lcs = star.get_lc(mission, author, cadence_period, n_batches)

        # Iterate over the light curves
        for lc in lcs:
            # Assign exoplanets and config string
            lc.exoplanets = star.exoplanets
            lc.config_str = f"{star.primary_name},{mission},{author},{cadence_period},batch={lc.batch}"
            # Save the light curve
            lc.save_to_file(dst_path / f"{lc.config_str}.json")
