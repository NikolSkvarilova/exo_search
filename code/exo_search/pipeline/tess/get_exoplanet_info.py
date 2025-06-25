import pandas as pd
from pathlib import Path


def main(filepath: Path, dst_filepath: Path):
    """Parse the TESS project candidates data to get information about exoplanets.

    Args:
        filepath (Path): file path to the CSV file.
        dst_filepath (Path): file path for the new exoplanet data.
    """
    # Read the data
    df = pd.read_csv(filepath)

    # Keep only relevant columns
    exoplanet_df = (
        df[["toi", "tid", "tfopwg_disp", "pl_orbper", "pl_tranmid", "pl_trandurh"]]
        .reset_index()
        .drop(columns="index")
    )

    # Rename columns
    exoplanet_df["star_name"] = exoplanet_df["tid"].apply(lambda x: f"TIC {x}")
    exoplanet_df["name"] = exoplanet_df["toi"].apply(lambda x: f"TOI {x}")
    exoplanet_df = exoplanet_df.rename(
        columns={
            "pl_orbper": "period",
            "pl_tranmid": "transit_midpoint_BJD",
            "tfopwg_disp": "disposition",
            "pl_trandurh": "transit_duration_h",
        }
    )
    exoplanet_df = exoplanet_df.drop(columns=["toi", "tid"])
    exoplanet_df = exoplanet_df[
        [
            "name",
            "star_name",
            "disposition",
            "period",
            "transit_midpoint_BJD",
            "transit_duration_h",
        ]
    ]

    # Save data
    dst_filepath.parent.mkdir(parents=True, exist_ok=True)
    exoplanet_df.to_csv(dst_filepath, index=False)
