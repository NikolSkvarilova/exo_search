import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import scienceplots
import numpy as np
from typing import List

plt.style.use(["science", "no-latex"])


def group_by_star(df: pd.DataFrame, col: str, ascending: bool = False):
    return (
        df.sort_values(col, ascending=ascending)
        .groupby("star_name", as_index=False)
        .first()
        .sort_values(col, ascending=ascending)
    )

def main(path: Path, dst_path: Path, plot_path: Path, bad_models_filepath: Path = None, file_format: str = "png"):
    dst_path.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)

    filenames = [fn for fn in path.iterdir() if ".csv" in fn.name]

    bad_models = []
    if bad_models_filepath is not None:
        bad_models_df = pd.read_csv(bad_models_filepath)
        if not bad_models_df.empty:
            bad_models = list(bad_models_df["bad_models"])

    # load the results, and put the transit model name from the filename as a column
    results_dfs = []
    for fn in filenames:
        df = pd.read_csv(fn)
        df["transit_model"] = fn.name[:-4]  # remove the ".csv" from the name
        results_dfs.append(df)

    # concatenate the results into a single dataframe
    results_df = pd.concat(results_dfs)

    # Remove bad models
    mask = results_df["transit_model"].isin(bad_models)
    results_df = results_df[~mask]

    # (a) group by count
    grouped_count = (
        results_df.sort_values("count", ascending=False)
        .groupby("star_name", as_index=False)
        .first()
        .sort_values("count", ascending=False)
    )

    grouped_count[["star_name", "count", "transit_model"]].to_csv(
        dst_path / "top_count.csv", index=False
    )

    # (b) corr_threshold_mean
    grouped_corr_threshold_mean = (
        results_df.sort_values(by="corr_threshold_mean", ascending=False)
        .groupby("star_name", as_index=False)
        .first()
        .sort_values("corr_threshold_mean", ascending=False)
    )

    grouped_corr_threshold_mean["corr_threshold_mean"] = round(
        grouped_corr_threshold_mean["corr_threshold_mean"], 4
    )

    grouped_corr_threshold_mean[
        ["star_name", "corr_threshold_mean", "transit_model"]
    ].to_csv(dst_path / "top_corr_threshold_mean.csv", index=False)


    # list how many stars have how many transits detected
    grouped_count["n_stars"] = 1
    grouped_count_by_categories = (
        grouped_count.groupby("count").sum("n_stars")["n_stars"].reset_index()
    )

    # bin the data star counts into categories, and plot the data
    bins = [-1, 0, 1, 2, 3, 5, 10, 15, 25, 50, float("inf")]
    labels = [
        "0",
        "1",
        "2",
        "3",
        "4 - 5",
        "6 - 10",
        "11 - 15",
        "16 - 25",
        "26 - 50",
        "50+",
    ]
    grouped_count_by_categories["bins"] = pd.cut(
        grouped_count_by_categories["count"],
        bins=bins,
        labels=labels,
        right=True,
    )

    grouped_count_by_categories = (
        grouped_count_by_categories.groupby("bins", observed=True)
        .sum("n_stars")
        .reset_index()
    )

    plot_pie(grouped_count_by_categories, plot_path, file_format)

    # (c) Longest mean period
    # Filter only 4 and more
    mask = results_df["count"] >= 4
    results_df = results_df[mask]

    if not results_df.empty:
        grouped_period_mean = group_by_star(results_df, "period_mean")
        grouped_period_mean["period_mean"] = round(grouped_period_mean["period_mean"], 4)
        grouped_period_mean[["star_name", "period_mean", "transit_model", "count"]].to_csv(
            dst_path / "top_period_mean_4.csv", index=False
        )


def plot_pie(grouped_count_by_categories, plot_path, file_format):
    plt.figure(dpi=300)
    colormap = cm.get_cmap("GnBu")
    colors = colormap(np.linspace(0, 0.8, len(grouped_count_by_categories["bins"])))
    plt.title("Stars per transit count")
    _, _, labels = plt.pie(
        grouped_count_by_categories["n_stars"],
        labels=grouped_count_by_categories["bins"],
        autopct=lambda x: "{:.0f}".format(
            x * sum(grouped_count_by_categories["n_stars"]) / 100
        ),
        colors=colors,
        wedgeprops={"linewidth": 0.5, "edgecolor": "black"},
    )

    for label in labels:
        label.set_fontsize(8)

    plt.savefig(plot_path / f"pie.{file_format}")
    plt.close()


# Example usage:

if __name__ == "__main__":
    main(
        Path("../../t/results/correlation/parsed_corr_results"),
        Path("../../t/results/correlation/evaluated_corr_results/"),
        Path("../..t/figures/correlation/stars-per-transit-count.pdf"),
        bad_models=[],
    )
