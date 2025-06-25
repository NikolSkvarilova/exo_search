import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple


def mask(df: pd.DataFrame) -> pd.DataFrame:
    """Mask out results for models over their star.

    Args:
        df (pd.DataFrame): dataframe.

    Returns:
        pd.DataFrame: dataframe with nans.
    """
    df_filtered = df.copy()

    for idx, row in df_filtered.iterrows():
        star_name = row["star_name"]
        for col in df_filtered.columns[5:]:
            if star_name in col:
                df_filtered.at[idx, col] = np.nan

    return df_filtered


def cap(
    df_ok_detected: pd.DataFrame, df_detectable: pd.DataFrame, df_detected: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cap the number of ok detected transits to detectable. There may be multiple detected transits in place of one.
    Number of removed transits from ok_detected is also subtracted from the total count of detected transits.

    Args:
        df_ok_detected (pd.DataFrame): ok detected.
        df_detectable (pd.DataFrame): detectable.
        df_detected (pd.DataFrame): detected.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: capped ok detected and detected.
    """

    ok_detected_data = df_ok_detected.iloc[:, 5:]
    detectable_data = df_detectable.iloc[:, 5:]
    detected_data = df_detected.iloc[:, 5:]

    ok_detected_capped = np.minimum(ok_detected_data.values, detectable_data.values)
    difference = ok_detected_data - ok_detected_capped
    detected_data = detected_data - difference

    df_ok_detected_capped = df_ok_detected.copy()
    df_ok_detected_capped.iloc[:, 5:] = ok_detected_capped
    df_detected_capped = df_detected.copy()
    df_detected_capped.iloc[:, 5:] = detected_data
    return df_ok_detected_capped, df_detected_capped


def squeeze(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Sum the dataframe.

    Args:
        df (pd.DataFrame): dataframe.
        column_name (str): name for the new column with summed valued.

    Returns:
        pd.DataFrame: new dataframe.
    """
    df_sum = df[df.columns[5:]].sum()
    df_sum = df_sum.reset_index().rename(
        columns={"index": "transit_model", 0: column_name}
    )
    return df_sum


def main(
    detectable_csv: Path, detected_csv: Path, ok_detected_csv: Path, result_csv: Path
) -> None:
    df_detectable = pd.read_csv(detectable_csv).drop_duplicates()
    df_detected = pd.read_csv(detected_csv).drop_duplicates()
    df_ok_detected = pd.read_csv(ok_detected_csv).drop_duplicates()


    # Cap values in ok_detected to not exceed detectable
    df_ok_detected, df_detected = cap(df_ok_detected, df_detectable, df_detected)

    # Mask of models detecting on its light curve
    df_detectable = mask(df_detectable)
    df_detected = mask(df_detected)
    df_ok_detected = mask(df_ok_detected)

    # Sum the dataframes
    df_detectable_sum = squeeze(df_detectable, "detectable")
    df_detected_sum = squeeze(df_detected, "detected")
    df_ok_detected_sum = squeeze(df_ok_detected, "ok_detected")

    # Merge dataframes
    df = df_detected_sum.merge(df_detectable_sum, on="transit_model")
    df = df.merge(df_ok_detected_sum, on="transit_model")

    # Calculate the statistics
    df["precision"] = df["ok_detected"] / df["detected"]
    df["recall"] = df["ok_detected"] / df["detectable"]
    df["f1_score"] = 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])

    df["detection_rate"] = df["detected"] / df["detectable"]

    sorted_df = (
        df.sort_values(by="f1_score", ascending=False)
        .reset_index()
        .drop(columns=["index"])
    )

    result_csv.parent.mkdir(parents=True, exist_ok=True)
    sorted_df.to_csv(result_csv, index=False)


# Example usage:

if __name__ == "__main__":
    main(
        Path("../../t/results/correlation/detectable.csv"),
        Path("../../t/results/correlation/detected.csv"),
        Path("../../t/results/correlation/ok_detected.csv"),
        Path("./../../t/evaluation/transit_models/evaluation_of_transit_models.csv"),
    )
