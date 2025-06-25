from exo_search.entities.star import Star
from exo_search.entities.exoplanet import Exoplanet
import pandas as pd
from pathlib import Path
import pandas as pd


def main(stars_csv: Path, star_path: Path, exoplanets_csv: Path = None) -> None:
    """Create star objects based on a csv.

    Args:
        stars_csv (Path): csv filename with star names.
        star_path (Path): directory for the newly created Star objects.
        exoplanets_csv (Path, optional): csv with information about exoplanets. Defaults to None.
    """

    # Read the exoplanets
    exoplanets_df = None
    if exoplanets_csv is not None:
        exoplanets_df = pd.read_csv(exoplanets_csv)

    # Read the star names
    df = pd.read_csv(stars_csv)

    # Iterate over each row
    for _, row in df.iterrows():
        star_name = row["star_name"]

        # Find matching exoplanets.
        exoplanets = []
        if exoplanets_df is not None:
            matching_exoplanets = exoplanets_df[exoplanets_df["star_name"] == star_name]
            for _, row in matching_exoplanets.iterrows():
                # Create Exoplanet object
                exoplanets.append(Exoplanet.from_pd_series(row))

        # Create new star, pass the new exoplanets
        star = Star(star_name, exoplanets=exoplanets)
        # Save the star
        star.save_to_file(star_path / f"{star_name}.json")
