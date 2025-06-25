import pandas as pd
from pathlib import Path


def main(
    filepath: Path, dst_filepath: Path, disposition: str = None, strict: bool = False
):
    """Create star list from TESS project candidates table.

    Args:
        filepath (Path): File path to TESS project candidates CSV.
        dst_filepath (Path): File path for the new star list.
        disposition (str, optional): Only stars with at least one exoplanet with this disposition will be chosen. All are included by default.
        strict (bool, optional): Only stars with all exoplanets matching the disposition will be chosen. Defaults to None.
    """
    # Read the CSV
    df = pd.read_csv(filepath)

    # Filter the CSV
    df_filtered = None
    if disposition is None:
        df_filtered = df
    elif strict:
        to_drop = df[df["tfopwg_disp"] != disposition]["tid"].unique()
        df_filtered = df[~df["tid"].isin(to_drop)]
    else:
        df_filtered = df[df["tfopwg_disp"] == disposition]

    # Get only star names
    df_unique = (
        df_filtered.drop_duplicates(subset=["tid"]).reset_index().drop(columns="index")
    )
    df_unique["star_name"] = df_unique["tid"].apply(lambda x: f"TIC {x}")

    # Save results
    dst_filepath.parent.mkdir(parents=True, exist_ok=True)
    df_unique["star_name"].to_csv(dst_filepath, index=False)
