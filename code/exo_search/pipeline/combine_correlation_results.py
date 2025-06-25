import pandas as pd
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import List

def main(results_paths: List[Path], dst_path: Path):
    """Combine results from correlation.

    Args:
        results_paths (List[Path]): directory with the results.
        dst_path (Path): directory for the combined results.
    """

    # create the destination path for the csv files
    dst_path.mkdir(parents=True, exist_ok=True)

    # all filenames with results
    filepaths_groups = []
    for results_path in results_paths:
        filepaths_groups.append([fn for fn in results_path.iterdir()])

    # get star_names
    star_names = set()
    for filepaths in filepaths_groups:
        for fn in filepaths:
            if "lc=" in fn.name:
                start = fn.name.index("lc=") + 3
                lc_info = fn.name[start:]
                star_name = lc_info.split(",")[0]
                star_names.add(star_name)

    transit_names = set()
    lcs_all = defaultdict(list)
    for star_name in star_names:
        # load all results from the same star
        selected_filepaths = [
            fn
            for filepaths in filepaths_groups
            for fn in filepaths
            if star_name in fn.name
        ]

        # lcs
        for fn in selected_filepaths:
            with open(fn, "r") as f:
                loaded = json.load(f)
                lcs_all[star_name].append(loaded[0])
        # unique transit names
        for lc in lcs_all[star_name]:
            for key, item in lc["t_positions"].items():
                transit_names.add(key)

        break

    all_rows = defaultdict(list)

    star_count = 1
    stars_total = len(star_names)
    for star_name in star_names:
        print(f"> Processing star: {star_count} / {stars_total}")
        star_count += 1
        selected_filepaths = [
            fn
            for filepaths in filepaths_groups
            for fn in filepaths
            if star_name in fn.name
        ]

        lcs = []
        for fn in selected_filepaths:
            with open(fn, "r") as f:
                loaded = json.load(f)
                lcs.append(loaded[0])

        # lcs contains results for all light curves from a single star
        # now select results from each light curve for the specific transit

        for transit_name in transit_names:
            transit_times_for_star = []
            corr_threshold_for_star = []

            for lc in lcs:
                transit_times = lc["t_positions"].get(transit_name, None)
                corr_threshold = lc["corr_threshold"].get(transit_name, None)

                if transit_times is not None and corr_threshold is not None:
                    transit_times_for_star.extend(transit_times)
                    corr_threshold_for_star.extend(corr_threshold)

            # if no data is found, probably the transit was not correlated with this star
            if transit_times_for_star == []:
                print(
                    f">> Transit {transit_name} was not correlated with star {star_name}"
                )
                continue

            transit_times_for_star_detected = (
                []
            )  # values from transit_times_for_star with corresponding value from corr_threshold_for_star larger than 1
            corr_threshold_for_star_detected = (
                []
            )  # values from corr_threshold_for_star larger than 1

            for index, corr_threshold_value in enumerate(corr_threshold_for_star):
                if corr_threshold_value >= 1:
                    transit_times_for_star_detected.append(
                        transit_times_for_star[index]
                    )
                    corr_threshold_for_star_detected.append(
                        corr_threshold_for_star[index]
                    )

            row = {"star_name": star_name}

            # check if some transits were found
            if len(transit_times_for_star_detected) > 0:
                # transits were detected, store the count, mean, std, etc.

                # sort before
                indexes = np.array(transit_times_for_star_detected).argsort()
                transit_times_for_star_detected = np.array(
                    transit_times_for_star_detected
                )[indexes]
                corr_threshold_for_star_detected = np.array(
                    corr_threshold_for_star_detected
                )[indexes]

                detected_transits_count = len(transit_times_for_star_detected)
                row["count"] = detected_transits_count

                # calculate the mean and std of the corr_thresholds
                row["corr_threshold_mean"] = np.mean(
                    corr_threshold_for_star_detected
                ).item()
                row["corr_threshold_std"] = np.std(
                    corr_threshold_for_star_detected
                ).item()

                # if the transit count exceeds 3, calculate information about periodicity
                if detected_transits_count >= 3:
                    # calculate periodicity information
                    distances_between_transits = []
                    distances_between_transits_scaled = (
                        []
                    )  # contains distances between transits divided by the index of the transit

                    for i, transit_time_1 in enumerate(
                        transit_times_for_star_detected[:-1]
                    ):
                        for j in range(i + 1, len(transit_times_for_star_detected)):
                            transit_time_2 = transit_times_for_star_detected[j]
                            distance = transit_time_2 - transit_time_1
                            distances_between_transits.append(distance)
                            distances_between_transits_scaled.append(distance / (j - i))

                    row["period_mean"] = np.mean(
                        distances_between_transits_scaled
                    ).item()
                    row["period_std"] = np.std(distances_between_transits_scaled).item()

            else:
                # if no transit was detected, log None values and store the highest corr_threshold value
                row["count"] = 0
                row["corr_threshold_max"] = np.max(corr_threshold_for_star).item()

            all_rows[transit_name].append(row)

    results_count = 1
    results_total = len(transit_names)

    # create and save the dataframe
    for transit_name, row_collection in all_rows.items():
        print(f"Saving results {results_count} / {results_total} ({transit_name})")
        results_count += 1
        df = pd.DataFrame(row_collection)
        df.to_csv(dst_path / f"{transit_name}.csv", index=False)


# Example usage:

if __name__ == "__main__":
    main(
        [
            Path("../../t/results/correlation/corr_results/"),
        ],
        Path("../../t/results/correlation/parsed_corr_results"),
    )
